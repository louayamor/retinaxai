import React from 'react';
import Image from 'next/image';
import { SidebarTrigger } from '../ui/sidebar';
import { Separator } from '../ui/separator';
import { Breadcrumbs } from '../breadcrumbs';
import SearchInput from '../search-input';
import { UserNav } from './user-nav';
import { ConnectionStatus } from '@/components/layout/connection-status';
import { NotificationCenter } from '@/components/notifications/notification-center';

export default function Header() {
  return (
    <header className='flex h-20 shrink-0 items-center justify-between gap-4 transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-16 bg-sidebar text-sidebar-foreground px-4'>
      <div className='flex items-center gap-2'>
        <SidebarTrigger className='h-8 w-8' />
        <Separator orientation='vertical' className='h-5' />
        <Breadcrumbs />
      </div>

      <div className='flex items-center gap-2'>
        <div className='hidden md:flex'>
          <SearchInput />
        </div>
        <ConnectionStatus />
        <NotificationCenter />
        <Image
          src='https://www.samayahospital.ae/home/images/logo.png'
          alt='Samaya Specialized Center'
          width={40}
          height={40}
          className='h-10 w-auto'
          unoptimized
        />
        <UserNav />
      </div>
    </header>
  );
}
