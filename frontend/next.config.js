/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    // Backend origin — override with BACKEND_URL (e.g. http://localhost:8001).
    // Defaults to :8000 for backward compatibility.
    const backend = process.env.BACKEND_URL || "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backend}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
