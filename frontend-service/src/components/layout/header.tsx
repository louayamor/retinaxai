import React from 'react';
import Image from 'next/image';
import { SidebarTrigger } from '../ui/sidebar';
import { Separator } from '../ui/separator';
import SearchInput from '../search-input';
import { UserNav } from './user-nav';
import { NotificationCenter } from '@/components/notifications/notification-center';

export default function Header() {
  return (
    <header className="flex h-14 shrink-0 items-center justify-between gap-3 transition-[width,height] ease-linear bg-sidebar text-sidebar-foreground border-b border-sidebar-border px-3">
      <div className="flex items-center gap-2 min-w-0 flex-[2]">
        <SidebarTrigger className="h-5 w-5 shrink-0" />
        <Separator orientation="vertical" className="h-3.5 bg-sidebar-border shrink-0" />
        <div className="hidden md:flex flex-1 max-w-md">
          <SearchInput />
        </div>
      </div>

      <div className="flex items-center shrink-0">
        <Image
          src="https://www.samayahospital.ae/home/images/logo.png"
          alt="Samaya Specialized Center"
          width={80}
          height={24}
          className="h-8 w-auto object-contain opacity-80"
          unoptimized
        />
      </div>

      <div className="flex items-center gap-1.5 flex-1 justify-end min-w-0">
        <NotificationCenter />
        <UserNav />
      </div>
    </header>
  );
}