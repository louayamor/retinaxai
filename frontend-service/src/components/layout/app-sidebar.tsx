'use client';

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger
} from '@/components/ui/collapsible';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  SidebarRail
} from '@/components/ui/sidebar';
import { navGroups, navItems } from '@/config/nav-config';
import { useFilteredNavItems } from '@/hooks/use-nav';
import { useMediaQuery } from '@/hooks/use-media-query';
import { clearTokens, apiFetch } from '@/lib/auth';
import {
  IconBell,
  IconChevronRight,
  IconChevronsDown,
  IconLogout,
  IconUserCircle
} from '@tabler/icons-react';
import Image from 'next/image';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import * as React from 'react';
import { Icons } from '../icons';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';

function parseUserFromToken(token: string | null) {
  if (!token) return null;
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    return { username: payload.sub ?? 'Doctor', email: payload.email ?? '' };
  } catch {
    return null;
  }
}

function NavItemComponent({
  item,
  pathname,
}: {
  item: typeof navItems[number];
  pathname: string;
}) {
  const Icon = item.icon ? Icons[item.icon] : Icons.logo;

  if (item.items && item.items.length > 0) {
    return (
      <Collapsible
        key={item.title}
        asChild
        defaultOpen={pathname.startsWith(item.url)}
        className='group/collapsible'
      >
        <SidebarMenuItem>
          <CollapsibleTrigger asChild>
            <SidebarMenuButton
              tooltip={item.title}
              isActive={pathname === item.url}
              className='text-sm py-1.5'
            >
              {item.icon && <Icon className="size-4" />}
              <span>{item.title}</span>
              <IconChevronRight className='ml-auto transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90' />
            </SidebarMenuButton>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <SidebarMenuSub>
              {item.items.map((subItem) => (
                <SidebarMenuSubItem key={subItem.title}>
                    <SidebarMenuSubButton
                      asChild
                      isActive={pathname === subItem.url}
                      className='text-sm py-1.5'
                    >
                    <Link href={subItem.url}>
                      <span>{subItem.title}</span>
                    </Link>
                  </SidebarMenuSubButton>
                </SidebarMenuSubItem>
              ))}
            </SidebarMenuSub>
          </CollapsibleContent>
        </SidebarMenuItem>
      </Collapsible>
    );
  }

  return (
    <SidebarMenuItem key={item.title}>
      <SidebarMenuButton
        asChild
        tooltip={item.title}
        isActive={pathname === item.url}
        className='text-sm py-1.5'
      >
        <Link href={item.url}>
          <Icon className="size-4" />
          <span>{item.title}</span>
        </Link>
      </SidebarMenuButton>
    </SidebarMenuItem>
  );
}

export default function AppSidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const filteredItems = useFilteredNavItems(navItems);
  const [user, setUser] = React.useState<{ username: string; email: string } | null>(null);

  React.useEffect(() => {
    apiFetch<{ username: string; email: string }>('/api/v1/auth/me')
      .then(setUser)
      .catch(() => setUser(null));
  }, []);

  async function handleLogout() {
    clearTokens();

    try {
      await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/auth/logout`, {
        method: 'POST',
        credentials: 'include',
      });
    } catch {}

    window.location.href = '/auth/login';
  }

  return (
    <Sidebar collapsible='icon'>
      <SidebarHeader>
        <div className='flex items-center gap-2 px-2 py-1.5'>
          <Image
            src='/retinaxai-logo.svg'
            alt='RetinaXAI'
            width={100}
            height={100}
            className='h-8 w-8 rounded-md'
            priority
          />
          <span className='font-semibold transition-opacity duration-200 group-data-[collapsible=icon]:opacity-0'>
            RetinaXAI
          </span>
        </div>
      </SidebarHeader>

      <SidebarContent>
        {navGroups.map((group) => (
          <SidebarGroup key={group.title}>
            <SidebarGroupLabel className='text-[10px] uppercase tracking-wider text-muted-foreground px-2'>
              {group.title}
            </SidebarGroupLabel>
            <SidebarMenu>
              {group.items.map((item) => (
                <NavItemComponent key={item.title} item={item} pathname={pathname} />
              ))}
            </SidebarMenu>
          </SidebarGroup>
        ))}
      </SidebarContent>

      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size='lg'
                  className='data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground'
                >
                  <Avatar className='h-7 w-7 rounded-md'>
                    <AvatarFallback className='rounded-md'>
                      {user?.username?.slice(0, 2).toUpperCase() ?? 'DR'}
                    </AvatarFallback>
                  </Avatar>
                  <div className='grid flex-1 text-left text-sm leading-tight'>
                    <span className='truncate font-semibold'>{user?.username ?? 'Doctor'}</span>
                    <span className='truncate text-xs text-muted-foreground'>{user?.email ?? ''}</span>
                  </div>
                  <IconChevronsDown className='ml-auto size-3' />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                className='w-(--radix-dropdown-menu-trigger-width) min-w-56 rounded-lg'
                side='bottom'
                align='end'
                sideOffset={4}
              >
                <DropdownMenuLabel className='p-0 font-normal'>
                  <div className='px-1 py-1.5'>
                    <p className='text-sm font-medium'>{user?.username}</p>
                    <p className='text-xs text-muted-foreground'>{user?.email}</p>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuGroup>
                  <DropdownMenuItem onClick={() => router.push('/dashboard/profile')}>
                    <IconUserCircle className='mr-2 h-4 w-4' />
                    Profile
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    <IconBell className='mr-2 h-4 w-4' />
                    Notifications
                  </DropdownMenuItem>
                </DropdownMenuGroup>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleLogout}>
                  <IconLogout className='mr-2 h-4 w-4' />
                  Log out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
