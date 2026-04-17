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
        title: 'OCT Analytics',
        url: '/dashboard/visualise',
        icon: 'chart',
        isActive: false,
        shortcut: ['v', 'v'],
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
    ],
  },
  {
    title: 'AI Tools',
    items: [
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
            shortcut: ['p', 'p'],
          },
          {
            title: 'GradCAM',
            url: '/dashboard/predictions/gradcam',
            icon: 'media',
            shortcut: ['g', 'g'],
          },
        ],
      },
      {
        title: 'Clinical Reports',
        url: '/dashboard/reports',
        icon: 'post',
        isActive: false,
        shortcut: ['r', 'r'],
        items: [],
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
