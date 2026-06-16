import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  devIndicators: false,
  output: 'standalone',
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'Content-Security-Policy',
            value: [
              "default-src 'self'",
              "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
              "style-src 'self' 'unsafe-inline'",
              "img-src 'self' data: blob: https: http://localhost:8000",
              "font-src 'self' data:",
              "connect-src 'self' http://localhost:8000 ws://localhost:8000 ws: wss:",
              "frame-src 'self'",
              "manifest-src 'self'",
              "media-src 'self'",
              "object-src 'none'",
              "base-uri 'self'",
              "form-action 'self'",
            ].join('; '),
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
        ],
      },
    ];
  },
  async rewrites() {
    return [
      {
        source: '/api/v1/system/grafana/proxy/:path*',
        destination: 'http://localhost:8000/api/v1/system/grafana/proxy/:path*',
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
        protocol: 'http',
        hostname: 'localhost',
        port: '8000'
      },
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
