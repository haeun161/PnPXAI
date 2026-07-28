"""Load a time-series forecaster from whatever the user supplied.

The task is forecast-only but no longer architecture-only: the bundled samples run
iTransformer, while an uploaded or linked model can be anything. Every source ends at the
same place — a `ForecasterWrapper` exposing the small interface the explanation pipeline
depends on:

    SEQ_LEN / PRED_LEN      the window the weights are pinned to
    forward(x)              one scalar per item, so the classification-shaped explainers
                            can attribute "which input timesteps drove the next value"
    forecast_horizon(ctx)   the whole (pred_len, channels) horizon, for the chart

`load_forecaster` recognises five sources and tries them in this order, because the cheap
unambiguous checks have to run before the ones that guess:

  1. TorchScript archive              torch.jit.save(model)
  2. pickled nn.Module                torch.save(model)          — any architecture
  3. {"state_dict", "config"}         self-describing iTransformer
  4. bare reference iTransformer state_dict
  5. HuggingFace repo id              transformers *ForPrediction* (PatchTST, Informer, …)

Values are passed in raw units throughout. Models that normalise internally (iTransformer's
instance norm, PatchTST's scaler) undo any dataset-level scaling themselves, so training-set
statistics are never needed here.
"""
from pathlib import Path

import torch

# Attribute names the wild uses for the two window lengths. Checked on the module, and on
# a `configs`/`config`/`args` holder hanging off it, which is where most reference
# implementations park their hyperparameters.
_SEQ_ATTRS = ("seq_len", "context_length", "input_len", "input_length", "lookback",
              "lookback_len", "window_size", "n_steps_in")
_PRED_ATTRS = ("pred_len", "prediction_length", "horizon", "out_len", "output_len",
               "forecast_len", "n_steps_out")

# Window lengths to try when the module says nothing about its own. Ordered by how common
# they are as forecasting benchmarks, so the first hit is usually the intended one.
_SEQ_CANDIDATES = (96, 512, 336, 192, 104, 60, 48, 36, 24, 12)


def _tensor_from(output):
    """First tensor in whatever a model returned (tensor, tuple, or HF ModelOutput)."""
    if isinstance(output, torch.Tensor):
        return output
    for key in ("prediction_outputs", "prediction_logits", "logits", "last_hidden_state"):
        value = getattr(output, key, None)
        if isinstance(value, torch.Tensor):
            return value
    if isinstance(output, (tuple, list)):
        return next((o for o in output if isinstance(o, torch.Tensor)), None)
    if isinstance(output, dict):
        return next((v for v in output.values() if isinstance(v, torch.Tensor)), None)
    return None


def _find_length(model, names):
    for holder in (model, getattr(model, "configs", None), getattr(model, "config", None),
                   getattr(model, "args", None)):
        if holder is None:
            continue
        for name in names:
            value = getattr(holder, name, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


class ForecasterWrapper(torch.nn.Module):
    """Uniform interface over one loaded forecaster.

    Subclasses implement `_predict`, which takes (batch, seq_len, channels) and returns
    (batch, pred_len, channels) — the layout the reference forecasting implementations
    agree on. Everything the pipeline calls is built on top of that here, once.
    """

    #: Human-readable architecture, surfaced in errors and in the model picker.
    kind = "forecaster"

    def __init__(self, model, seq_len: int, pred_len: int, num_channels=None):
        super().__init__()
        self._model = model
        self._model.eval()
        self.SEQ_LEN = int(seq_len)
        self.PRED_LEN = int(pred_len)
        #: Channel count the weights are fixed to, or None when the architecture is
        #: channel-agnostic (iTransformer embeds each variate independently, so one
        #: checkpoint runs on any number of columns; PatchTST does not).
        self.NUM_CHANNELS = num_channels

    def _predict(self, x_enc: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def set_time_context(self, timestamps) -> None:
        """Attach calendar covariates for the current context window. Only the reference
        iTransformer consumes them; for everyone else this is deliberately a no-op."""

    @staticmethod
    def channel_for(num_channels: int) -> int:
        """The forecast target is the last column of the upload, by convention — which is
        where ETT-style CSVs put it (OT) and where the trained checkpoints put the label.
        Deriving it from the data rather than from the weights keeps any dataset working
        without per-file configuration."""
        return max(num_channels - 1, 0)

    def _fit_window(self, x: torch.Tensor) -> torch.Tensor:
        """Trim/pad the time axis to exactly SEQ_LEN (the weights fix this length)."""
        seq_len = x.shape[-1]
        if seq_len > self.SEQ_LEN:
            return x[..., -self.SEQ_LEN:]
        if seq_len < self.SEQ_LEN:
            pad = x[..., :1].expand(*x.shape[:-1], self.SEQ_LEN - seq_len)
            return torch.cat([pad, x], dim=-1)
        return x

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, seq_len) -> (batch, pred_len, channels)."""
        if self.NUM_CHANNELS is not None and x.shape[1] != self.NUM_CHANNELS:
            raise ValueError(
                f"This {self.kind} was trained on {self.NUM_CHANNELS} variables but the "
                f"data has {x.shape[1]}. Unlike iTransformer, it embeds a fixed set of "
                f"channels, so the column count has to match."
            )
        return self._predict(self._fit_window(x).transpose(1, 2))

    def forward(self, x):
        # Single always-selected pseudo-class, so softmax/argmax downstream stay valid.
        out = self._run(x)
        ch = self.channel_for(out.shape[-1])
        return out[:, 0, ch:ch + 1]

    @torch.no_grad()
    def forecast_horizon(self, context: torch.Tensor) -> torch.Tensor:
        """context: (channels, seq_len) or (1, channels, seq_len) -> (pred_len, channels)."""
        if context.dim() == 2:
            context = context.unsqueeze(0)
        return self._run(context)[0]


class _ITransformerReference(ForecasterWrapper):
    """The reference implementation: a bare state_dict, and it wants calendar covariates
    alongside the variate tokens."""

    kind = "iTransformer"

    def __init__(self, model, seq_len, pred_len):
        super().__init__(model, seq_len, pred_len)
        self._x_mark = None

    def set_time_context(self, timestamps) -> None:
        """Calendar features for the context window, or None to drop them.

        The reference training pipeline picks the feature set from the data's sampling
        frequency (gluonts convention: weekly -> [day-of-month, week-of-year], daily -> 3,
        hourly -> 4). The checkpoint consumed exactly that set during training, so it is
        re-derived from the timestamp spacing — handing an hourly feature set to a
        weekly-trained model would feed it covariate tokens it has never seen.
        """
        import numpy as np
        import pandas as pd
        if timestamps is None:
            self._x_mark = None
            return
        step = (timestamps[-1] - timestamps[0]) / max(len(timestamps) - 1, 1)
        day_of_week = timestamps.dayofweek / 6.0 - 0.5
        day_of_month = (timestamps.day - 1) / 30.0 - 0.5
        day_of_year = (timestamps.dayofyear - 1) / 365.0 - 0.5
        if step >= pd.Timedelta(days=6):        # weekly (freq='w')
            week = (np.asarray(timestamps.isocalendar().week, dtype=np.float64) - 1) / 52.0 - 0.5
            columns = [day_of_month, week]
        elif step >= pd.Timedelta(hours=23):    # daily (freq='d')
            columns = [day_of_week, day_of_month, day_of_year]
        else:                                   # hourly and finer (freq='h')
            columns = [timestamps.hour / 23.0 - 0.5, day_of_week, day_of_month, day_of_year]
        feats = np.stack(columns, axis=1).astype(np.float32)
        self._x_mark = torch.from_numpy(feats).unsqueeze(0)

    def _predict(self, x_enc):
        x_mark = None
        if self._x_mark is not None:
            x_mark = self._x_mark.to(x_enc.device, x_enc.dtype).expand(x_enc.shape[0], -1, -1)
        return self._model(x_enc, x_mark, None, None)


class _ITransformerConfigured(ForecasterWrapper):
    """Our own {state_dict, config} export — same architecture, no covariates."""

    kind = "iTransformer"

    def _predict(self, x_enc):
        return self._model(x_enc)


class _HuggingFaceForecaster(ForecasterWrapper):
    """A transformers `*ForPrediction` model (PatchTST, PatchTSMixer, Informer, …).

    They agree on taking `past_values` as (batch, context, channels) and returning the
    horizon in the same layout, which is exactly `_predict`'s contract. The families that
    also demand calendar covariates (TimeSeriesTransformer, Autoformer, Informer) are
    rejected at load time rather than fed zeros, since a covariate the model was trained
    to rely on is not something to silently fabricate.
    """

    kind = "HuggingFace forecaster"

    def _predict(self, x_enc):
        out = _tensor_from(self._model(past_values=x_enc))
        if out is None:
            raise RuntimeError("The model returned no tensor to read a forecast from.")
        # Probabilistic heads emit (batch, samples, pred_len, channels); collapse the
        # sample axis to its mean so the chart and the attribution see one trajectory.
        if out.dim() == 4:
            out = out.mean(dim=1)
        return out


class _PickledModule(ForecasterWrapper):
    """A whole nn.Module the user saved with torch.save(model) or torch.jit.save.

    The architecture is unknown, so both the input layout and the window lengths are
    established once at load time by actually running the thing (see `_probe`), and the
    layout is replayed here.
    """

    kind = "model"

    def __init__(self, model, seq_len, pred_len, num_channels, channels_first: bool):
        super().__init__(model, seq_len, pred_len, num_channels=num_channels)
        self._channels_first = channels_first

    def _predict(self, x_enc):
        out = _tensor_from(self._model(x_enc.transpose(1, 2) if self._channels_first else x_enc))
        if out is None:
            raise RuntimeError("The model returned no tensor to read a forecast from.")
        if out.dim() == 2:                      # (batch, pred_len) — target channel only
            out = out.unsqueeze(-1)
        if self._channels_first and out.shape[1] == x_enc.shape[-1]:
            out = out.transpose(1, 2)           # (batch, channels, pred) -> (batch, pred, channels)
        return out


def _infer_reference_config(state_dict: dict) -> dict:
    """Recover the reference iTransformer's layer sizes from the weights themselves.

    A bare state_dict carries no config, and the horizon differs per checkpoint (ETTh1 was
    trained 96->96, another dataset may be 12->3), so hardcoding a size would silently
    reject every checkpoint but one. Every dimension except the head count is pinned by a
    weight shape; heads only re-split an unchanged d_model matrix, so it stays at the
    reference default of 8.
    """
    embed = state_dict["enc_embedding.value_embedding.weight"]   # (d_model, seq_len)
    projector = state_dict["projector.weight"]                   # (pred_len, d_model)
    d_ff = state_dict["encoder.attn_layers.0.conv1.weight"].shape[0]
    e_layers = 1 + max(
        int(k.split(".")[2]) for k in state_dict if k.startswith("encoder.attn_layers.")
    )
    return {"seq_len": embed.shape[1], "pred_len": projector.shape[0],
            "d_model": embed.shape[0], "d_ff": d_ff, "e_layers": e_layers, "n_heads": 8}


# Variable counts to fall back on when the caller's guess is refused. Explainer detection
# runs before any data is chosen and asks with a placeholder count, so a model pinned to
# some other width has to be discoverable without being told.
_CHANNEL_CANDIDATES = (7, 1, 6, 8, 21, 321)


def _probe_shape(model, num_channels: int, declared_seq, declared_pred):
    """One channel count's worth of the search in `_probe`."""
    candidates = ([declared_seq] if declared_seq else
                  []) + [c for c in _SEQ_CANDIDATES if c != declared_seq]
    for seq_len in candidates:
        for channels_first in (False, True):
            shape = ((1, num_channels, seq_len) if channels_first else (1, seq_len, num_channels))
            try:
                with torch.no_grad():
                    out = _tensor_from(model(torch.zeros(*shape)))
            except Exception:
                continue
            if out is None or out.dim() < 2:
                continue
            # The horizon is whichever output axis isn't the channel axis. When both are
            # the same size the layouts are indistinguishable, and the model's own
            # declaration is the only thing that can break the tie.
            if out.dim() == 2:
                pred_len = out.shape[1]
            elif channels_first and out.shape[1] == num_channels:
                pred_len = out.shape[2]
            elif out.shape[-1] == num_channels:
                pred_len = out.shape[1]
            else:
                pred_len = declared_pred or out.shape[1]
            return seq_len, int(declared_pred or pred_len), channels_first, num_channels
    return None


def _probe(model, num_channels: int):
    """Discover how to call an unknown module, by calling it.

    Returns (seq_len, pred_len, channels_first, num_channels). Nothing about a pickled
    module is guaranteed — not the input layout, not the window it was trained for, not
    even the variable count — so each candidate combination gets one forward pass on zeros
    and the first that survives wins. Anything the module declares about itself is tried
    first, so a model that documents its own lengths is taken at its word rather than
    guessed at.
    """
    declared_seq = _find_length(model, _SEQ_ATTRS)
    declared_pred = _find_length(model, _PRED_ATTRS)
    counts = [num_channels] + [c for c in _CHANNEL_CANDIDATES if c != num_channels]
    for channels in counts:
        found = _probe_shape(model, channels, declared_seq, declared_pred)
        if not found:
            continue
        seq_len, pred_len, channels_first, _ = found
        # Whether the count that worked is a requirement or a coincidence: an
        # iTransformer-style model embeds each variate on its own and takes any width,
        # a PatchTST-style one is pinned to the width it was trained on. One more pass
        # at a different width is what separates the two, and it decides whether data
        # with a different column count gets rejected later.
        agnostic = _probe_shape(model, channels + 1, seq_len, declared_pred) is not None
        return seq_len, pred_len, channels_first, (None if agnostic else channels)
    return None


def load_from_file(path, num_channels: int = 7) -> ForecasterWrapper:
    """Load a checkpoint file: TorchScript, a pickled module, or an iTransformer."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Checkpoint not found at {path}.")

    try:
        scripted = torch.jit.load(str(path), map_location="cpu")
    except Exception:
        scripted = None
    if scripted is not None:
        probed = _probe(scripted, num_channels)
        if probed is None:
            raise RuntimeError(
                "This TorchScript model would not accept a (batch, time, channels) or "
                "(batch, channels, time) input, so its forecast window could not be "
                "determined."
            )
        seq_len, pred_len, channels_first, fixed_channels = probed
        return _PickledModule(scripted, seq_len, pred_len, fixed_channels, channels_first)

    # weights_only defaults to True on modern torch, which refuses a pickled module —
    # exactly the case this branch exists to support. The file is one the user chose to
    # upload, and loading it is the operation they asked for.
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    except ModuleNotFoundError as e:
        # torch.save(model) stores a *reference* to the class, not its source, so the
        # defining module has to be importable here — and a model trained in someone
        # else's project never is. TorchScript carries its own code and does not care.
        raise RuntimeError(
            f"This file was saved with torch.save(model), which records the class by name "
            f"({e.name}) instead of storing its code, and that module does not exist on "
            f"this server. Re-save it with torch.jit.save(torch.jit.trace(model, example)) "
            f"— TorchScript is self-contained — or as {{'state_dict': ..., 'config': ...}}."
        ) from e

    if isinstance(obj, torch.nn.Module):
        probed = _probe(obj, num_channels)
        if probed is None:
            raise RuntimeError(
                f"Loaded a {type(obj).__name__} module, but it would not accept a "
                f"(batch, time, {num_channels}) or (batch, {num_channels}, time) input, so "
                f"its forecast window could not be determined. Check the variable count, "
                f"or re-save it as {{'state_dict': ..., 'config': ...}}."
            )
        seq_len, pred_len, channels_first, fixed_channels = probed
        wrapper = _PickledModule(obj, seq_len, pred_len, fixed_channels, channels_first)
        wrapper.kind = type(obj).__name__
        return wrapper

    if isinstance(obj, dict) and "state_dict" in obj and "config" in obj:
        from backend.models.itransformer import ITransformerNet
        cfg = obj["config"]
        model = ITransformerNet(**cfg)
        model.load_state_dict(obj["state_dict"])
        return _ITransformerConfigured(model, cfg["seq_len"], cfg["pred_len"])

    if isinstance(obj, dict) and "enc_embedding.value_embedding.weight" in obj:
        from backend.models.itransformer import Model
        cfg = _infer_reference_config(obj)
        model = Model(**cfg)
        model.load_state_dict(obj)
        return _ITransformerReference(model, cfg["seq_len"], cfg["pred_len"])

    kind = type(obj).__name__
    detail = "a state_dict for some other architecture" if isinstance(obj, dict) else kind
    raise RuntimeError(
        f"Could not read this file as a forecaster — it holds {detail}. Supported: a "
        f"whole model saved with torch.save(model) or torch.jit.save, a file saved as "
        f"{{'state_dict': ..., 'config': ...}}, or a reference iTransformer state_dict. "
        f"A bare state_dict for another architecture carries no description of its own "
        f"layers, so it cannot be rebuilt."
    )


# Covariate-hungry families: they take past_time_features/future_time_features as
# required arguments, learned during training. Zeros there are not a neutral default.
_HF_NEEDS_COVARIATES = ("time_series_transformer", "autoformer", "informer")


def load_from_hub(repo_id: str) -> ForecasterWrapper:
    """Load a transformers `*ForPrediction` model from a HuggingFace repo id."""
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(repo_id)
    architectures = getattr(config, "architectures", None) or []
    arch = architectures[0] if architectures else ""

    if not arch.endswith("ForPrediction"):
        raise RuntimeError(
            f"'{repo_id}' is a {arch or config.model_type} checkpoint, which does not "
            f"forecast — this task needs a model with a prediction head (a *ForPrediction "
            f"checkpoint). Pretraining, masking and classification heads produce no horizon."
        )
    if config.model_type in _HF_NEEDS_COVARIATES:
        raise RuntimeError(
            f"'{repo_id}' is a {config.model_type} model, which requires the calendar "
            f"covariates it was trained with (past/future time features) as separate "
            f"inputs. Feeding it zeros would not reproduce its training conditions, so it "
            f"is not supported. PatchTST and PatchTSMixer checkpoints take raw values and "
            f"do work."
        )

    import transformers
    cls = getattr(transformers, arch, None) or AutoModel
    model = cls.from_pretrained(repo_id)

    seq_len = _find_length(config, _SEQ_ATTRS)
    pred_len = _find_length(config, _PRED_ATTRS)
    if not seq_len or not pred_len:
        raise RuntimeError(
            f"'{repo_id}' does not state its context and prediction lengths, so the input "
            f"window it expects is unknown."
        )
    num_channels = getattr(config, "num_input_channels", None)
    wrapper = _HuggingFaceForecaster(model, seq_len, pred_len, num_channels=num_channels)
    wrapper.kind = arch
    return wrapper


def load_forecaster(source, num_channels: int = 7) -> ForecasterWrapper:
    """Load from a checkpoint path or a HuggingFace repo id."""
    if isinstance(source, (str, Path)) and Path(source).exists():
        return load_from_file(source, num_channels=num_channels)
    return load_from_hub(str(source))
