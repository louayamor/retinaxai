'use client';
import { useKBar } from 'kbar';
import { IconSearch } from '@tabler/icons-react';
import { Button } from './ui/button';

export default function SearchInput() {
  const { query } = useKBar();
  return (
    <div className='w-full space-y-2'>
      <Button
        variant='outline'
        className='bg-sidebar text-sidebar-foreground border-sidebar-border hover:bg-sidebar-accent hover:text-sidebar-accent-foreground relative h-8 w-full justify-start rounded-md text-sm font-normal shadow-none sm:pr-12 md:w-40 lg:w-56'
        onClick={query.toggle}
      >
        <IconSearch className='mr-2 h-4 w-4' />
        Search...
        <kbd className='bg-sidebar-accent pointer-events-none absolute top-[0.25rem] right-[0.25rem] hidden h-5 items-center gap-1 rounded border border-sidebar-border px-1.5 font-mono text-[10px] font-medium opacity-100 select-none sm:flex'>
          <span className='text-xs'>⌘</span>K
        </kbd>
      </Button>
    </div>
  );
}
