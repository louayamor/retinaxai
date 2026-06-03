import type { NextConfig } from 'next';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const nextConfig: NextConfig = {
  devIndicators: false,
  output: 'standalone',
  async rewrites() {
    return [
      {
        source: '/api/v1/system/grafana/proxy/:path*',
        destination: `${API_URL}/api/v1/system/grafana/proxy/:path*`,
      },
    ];
  },
  async redirects() {
    return [
      { source: '/dashboard/overview/:path*', destination: '/dashboard/clinical/overview/:path*', permanent: true },
      { source: '/dashboard/overview', destination: '/dashboard/clinical', permanent: true },
      { source: '/dashboard/patients/:path*', destination: '/dashboard/clinical/patients/:path*', permanent: true },
      { source: '/dashboard/predictions/:path*', destination: '/dashboard/clinical/predictions/:path*', permanent: true },
      { source: '/dashboard/reports/:path*', destination: '/dashboard/clinical/reports/:path*', permanent: true },
      { source: '/dashboard/analytics/:path*', destination: '/dashboard/clinical/analytics/:path*', permanent: true },
      { source: '/dashboard/chat/:path*', destination: '/dashboard/clinical/chat/:path*', permanent: true },
      { source: '/dashboard/models/:path*', destination: '/dashboard/engineering/models/:path*', permanent: true },
      { source: '/dashboard/mlops/:path*', destination: '/dashboard/engineering/mlops/:path*', permanent: true },
      { source: '/dashboard/llmops/:path*', destination: '/dashboard/engineering/llmops/:path*', permanent: true },
      { source: '/dashboard/system/:path*', destination: '/dashboard/engineering/system/:path*', permanent: true },
    ];
  },
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'api.slingacademy.com',
        port: ''
      },
      {
        protocol: 'https',
        hostname: 'www.samayahospital.ae',
        port: ''
      },
      {
        protocol: 'https',
        hostname: 'images.unsplash.com',
        port: ''
      }
    ]
  }
};

export default nextConfig;
