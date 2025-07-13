/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  output: 'standalone',
  
  // Environment variables
  env: {
    WEATHER_API_URL: process.env.WEATHER_API_URL || 'http://localhost:8000',
    WEBSOCKET_URL: process.env.WEBSOCKET_URL || 'ws://localhost:8001',
  },
  
  // Image optimization
  images: {
    domains: ['openweathermap.org'],
    unoptimized: false,
  },
  
  // Headers for security and CORS
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Origin', value: '*' },
          { key: 'Access-Control-Allow-Methods', value: 'GET, POST, PUT, DELETE, OPTIONS' },
          { key: 'Access-Control-Allow-Headers', value: 'Content-Type, Authorization' },
        ],
      },
    ]
  },
  
  // Rewrites for API proxy
  async rewrites() {
    return [
      {
        source: '/api/weather/:path*',
        destination: `${process.env.WEATHER_API_URL || 'http://localhost:8000'}/:path*`,
      },
    ]
  },
  
  // Experimental features
  experimental: {
    serverComponentsExternalPackages: ['leaflet'],
  },
}

module.exports = nextConfig
