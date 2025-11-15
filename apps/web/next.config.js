/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'export',
  images: {
    unoptimized: true,
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'optcgapi.com',
      },
      {
        protocol: 'https',
        hostname: 'optcgap.com', // fallback for parallel images
      },
    ],
  },
};

module.exports = nextConfig;

