import { NavItem } from '@/types';

export const navItems: NavItem[] = [
  {
    title: 'Dashboard',
    url: '/dashboard/overview',
    icon: 'dashboard',
    isActive: false,
    shortcut: ['o', 'o'],
    items: []
  },
  {
    title: 'OCT Analytics',
    url: '/dashboard/visualise',
    icon: 'chart',
    isActive: false,
    shortcut: ['v', 'v'],
    items: []
  },
  {
    title: 'Patient Registry',
    url: '/dashboard/patients',
    icon: 'user',
    isActive: false,
    shortcut: ['p', 'p'],
    items: []
  },
  {
    title: 'DR Screening',
    url: '/dashboard/predictions',
    icon: 'media',
    isActive: false,
    shortcut: ['d', 'd'],
    items: [
      {
        title: 'Predictions',
        url: '/dashboard/predictions',
        icon: 'media',
        shortcut: ['p', 'p']
      },
      {
        title: 'GradCAM',
        url: '/dashboard/predictions/gradcam',
        icon: 'media',
        shortcut: ['g', 'g']
      }
    ]
  },
  {
    title: 'Clinical Reports',
    url: '/dashboard/reports',
    icon: 'post',
    isActive: false,
    shortcut: ['r', 'r'],
    items: []
  },
  {
    title: 'AI Models',
    url: '/dashboard/models',
    icon: 'settings',
    isActive: false,
    shortcut: ['m', 'm'],
    items: []
  },
  {
    title: 'MLOps Monitor',
    url: '/dashboard/mlops',
    icon: 'settings',
    isActive: false,
    shortcut: ['l', 'l'],
    items: []
  },
  {
    title: 'LLMOps Monitor',
    url: '/dashboard/llmops',
    icon: 'settings',
    isActive: false,
    shortcut: ['o', 'o'],
    items: []
  },
  {
    title: 'System Stats',
    url: '/dashboard/system',
    icon: 'settings',
    isActive: false,
    shortcut: ['s', 's'],
    items: []
  },
  {
    title: 'Account',
    url: '#',
    icon: 'account',
    isActive: true,
    items: [
      {
        title: 'Profile',
        url: '/dashboard/profile',
        icon: 'profile',
        shortcut: ['u', 'u']
      }
    ]
  }
];
