import { NavItem } from '@/types';

export type NavGroup = {
  title: string;
  items: NavItem[];
};

export const navGroups: NavGroup[] = [
  {
    title: '',
    items: [
      {
        title: 'Patients',
        url: '/dashboard/patients',
        icon: 'user',
        isActive: false,
        shortcut: ['p', 'p'],
        items: [],
      },
      {
        title: 'Predictions',
        url: '/dashboard/predictions',
        icon: 'media',
        isActive: false,
        shortcut: ['d', 'd'],
        items: [],
      },
      {
        title: 'Reports',
        url: '/dashboard/reports',
        icon: 'post',
        isActive: false,
        shortcut: ['r', 'r'],
        items: [],
      },
      {
        title: 'Analytics',
        url: '/dashboard/analytics',
        icon: 'chart',
        isActive: false,
        shortcut: ['a', 'a'],
        items: [],
      },
      {
        title: 'AI Chat',
        url: '/dashboard/chat',
        icon: 'message',
        isActive: false,
        shortcut: ['c', 'c'],
        items: [],
      },
    ],
  },
  {
    title: 'System',
    items: [
      {
        title: 'AI Models',
        url: '/dashboard/models',
        icon: 'cpu',
        isActive: false,
        shortcut: ['m', 'm'],
        items: [],
      },
      {
        title: 'MLOps Monitor',
        url: '/dashboard/mlops',
        icon: 'server',
        isActive: false,
        shortcut: ['l', 'l'],
        items: [],
      },
      {
        title: 'LLMOps Monitor',
        url: '/dashboard/llmops',
        icon: 'activity',
        isActive: false,
        shortcut: ['o', 'o'],
        items: [],
      },
      {
        title: 'System Stats',
        url: '/dashboard/system',
        icon: 'database',
        isActive: false,
        shortcut: ['s', 's'],
        items: [],
      },
    ],
  },
];

export const navItems: NavItem[] = navGroups.flatMap((group) => group.items);
