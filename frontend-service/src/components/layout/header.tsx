import React from 'react';
import Image from 'next/image';
import { SidebarTrigger } from '../ui/sidebar';
import { Separator } from '../ui/separator';
import SearchInput from '../search-input';
import { UserNav } from './user-nav';
import { ConnectionStatus } from '@/components/layout/connection-status';
import { NotificationCenter } from '@/components/notifications/notification-center';

export default function Header() {
  return (
    <header className="flex h-14 shrink-0 items-center justify-between gap-4 transition-[width,height] ease-linear bg-sidebar text-sidebar-foreground border-b border-sidebar-border px-4">
      <div className="flex items-center gap-2 flex-1">
        <SidebarTrigger className="h-7 w-7" />
        <Separator orientation="vertical" className="h-4 bg-sidebar-border" />
        <div className="hidden md:flex">
          <SearchInput />
        </div>
      </div>

      <div className="flex items-center gap-2 absolute left-1/2 -translate-x-1/2">
        <Image
          src="https://www.samayahospital.ae/home/images/logo.png"
          alt="Samaya Specialized Center"
          width={120}
          height={36}
          className="h-8 w-auto object-contain "
          unoptimized
        />
      </div>

      <div className="flex items-center gap-2 flex-1 justify-end">
        <ConnectionStatus />
        <NotificationCenter />
        <UserNav />
      </div>
    </header>
  );
}