import { NavItem } from '@/types';

export type NavGroup = {
  title: string;
  items: NavItem[];
};

export const navGroups: NavGroup[] = [
  {
    title: 'Clinical',
    items: [
      {
        title: 'Dashboard',
        url: '/dashboard/overview',
        icon: 'dashboard',
        isActive: false,
        shortcut: ['o', 'o'],
        items: [],
      },
      {
        title: 'Patient Registry',
        url: '/dashboard/patients',
        icon: 'user',
        isActive: false,
        shortcut: ['p', 'p'],
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
    ],
  },
  {
    title: 'AI Assistant',
    items: [
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
    title: 'AI Tools',
    items: [
      {
        title: 'Diagnostics',
        url: '/dashboard/predictions?tab=screening',
        icon: 'media',
        isActive: false,
        shortcut: ['d', 'd'],
        items: [
          {
            title: 'Screening',
            url: '/dashboard/predictions?tab=screening',
            icon: 'media',
            shortcut: ['s', 's'],
          },
          {
            title: 'Reports',
            url: '/dashboard/predictions?tab=reports',
            icon: 'post',
            shortcut: ['r', 'r'],
          },
          {
            title: 'GradCAM',
            url: '/dashboard/predictions?tab=gradcam',
            icon: 'media',
            shortcut: ['g', 'g'],
          },
        ],
      },
    ],
  },
  {
    title: 'Administration',
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
  {
    title: 'Account',
    items: [
      {
        title: 'Profile',
        url: '/dashboard/profile',
        icon: 'profile',
        isActive: false,
        shortcut: ['u', 'u'],
        items: [],
      },
    ],
  },
];

export const navItems: NavItem[] = navGroups.flatMap((group) => group.items);