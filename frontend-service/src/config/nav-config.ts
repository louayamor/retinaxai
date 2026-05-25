import { NavItem } from '@/types';

export const clinicalNav: NavItem[] = [
  {
    title: 'Overview',
    url: '/dashboard/clinical',
    icon: 'dashboard',
    isActive: false,
    items: [],
  },
  {
    title: 'Patients',
    url: '/dashboard/clinical/patients',
    icon: 'user',
    isActive: false,
    shortcut: ['p', 'p'],
    items: [],
  },
  {
    title: 'Predictions',
    url: '/dashboard/clinical/predictions',
    icon: 'media',
    isActive: false,
    shortcut: ['d', 'd'],
    items: [],
  },
  {
    title: 'Reports',
    url: '/dashboard/clinical/reports',
    icon: 'post',
    isActive: false,
    shortcut: ['r', 'r'],
    items: [],
  },
  {
    title: 'Analytics',
    url: '/dashboard/clinical/analytics',
    icon: 'chart',
    isActive: false,
    shortcut: ['a', 'a'],
    items: [],
  },
  {
    title: 'AI Chat',
    url: '/dashboard/clinical/chat',
    icon: 'message',
    isActive: false,
    shortcut: ['c', 'c'],
    items: [],
  },
];

export const engineeringNav: NavItem[] = [
  {
    title: 'Overview',
    url: '/dashboard/engineering',
    icon: 'dashboard',
    isActive: false,
    items: [],
  },
  {
    title: 'AI Models',
    url: '/dashboard/engineering/models',
    icon: 'cpu',
    isActive: false,
    shortcut: ['m', 'm'],
    items: [],
  },
  {
    title: 'MLOps Monitor',
    url: '/dashboard/engineering/mlops',
    icon: 'server',
    isActive: false,
    shortcut: ['l', 'l'],
    items: [],
  },
  {
    title: 'LLMOps Monitor',
    url: '/dashboard/engineering/llmops',
    icon: 'activity',
    isActive: false,
    shortcut: ['o', 'o'],
    items: [],
  },
  {
    title: 'System Stats',
    url: '/dashboard/engineering/system',
    icon: 'database',
    isActive: false,
    shortcut: ['s', 's'],
    items: [],
  },
];

export const adminNav: NavItem[] = [
  {
    title: 'Overview',
    url: '/dashboard/admin',
    icon: 'dashboard',
    isActive: false,
    items: [],
  },
  {
    title: 'Users',
    url: '/dashboard/admin/users',
    icon: 'user',
    isActive: false,
    shortcut: ['u', 'u'],
    items: [],
  },
  {
    title: 'Settings',
    url: '/dashboard/admin/settings',
    icon: 'settings',
    isActive: false,
    items: [],
  },
];
