'use client';

import { useCallback, useEffect, useState } from 'react';
import PageContainer from '@/components/layout/page-container';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { fetchNotifications, markNotificationsRead, markAllNotificationsRead } from '@/lib/api';
import type { NotificationItem } from '@/lib/api';

const NOTIFICATION_TYPE_BADGE: Record<string, 'default' | 'secondary' | 'destructive' | 'outline' | 'info' | 'warning' | 'success'> = {
  info: 'info',
  warning: 'warning',
  error: 'destructive',
  success: 'success',
  system: 'default',
};

function getTypeVariant(type: string) {
  const key = type.toLowerCase();
  return NOTIFICATION_TYPE_BADGE[key] ?? 'secondary';
}

function timeAgo(dateStr: string) {
  const seconds = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000);
  if (seconds < 60) return 'just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export default function AdminJournalPage() {
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const items = await fetchNotifications(false, 100, 0);
      setNotifications(items);
    } catch { /* ignore */ }
    setLoading(false);
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  async function handleMarkRead(id: string) {
    try {
      await markNotificationsRead([id]);
      setNotifications(prev => prev.map(n => n.id === id ? { ...n, read: true } : n));
    } catch { /* ignore */ }
  }

  async function handleMarkAllRead() {
    try {
      await markAllNotificationsRead();
      setNotifications(prev => prev.map(n => ({ ...n, read: true })));
    } catch { /* ignore */ }
  }

  const unreadCount = notifications.filter(n => !n.read).length;

  return (
    <PageContainer>
      <div className='flex flex-1 flex-col gap-6 min-h-0'>
        <div className='rounded-lg border bg-gradient-to-r from-slate-900 via-cyan-900 to-teal-900 p-5 text-white shadow-sm'>
          <div className='relative z-10 space-y-1'>
            <div className='flex items-center justify-between'>
              <div>
                <h1 className='text-xl font-bold tracking-tight'>Journal</h1>
                <p className='max-w-xl text-sm text-white/70'>
                  Platform notifications and system events
                </p>
              </div>
              {unreadCount > 0 && (
                <div className='flex items-center gap-3'>
                  <Badge variant='destructive' className='h-6 px-2 text-xs'>
                    {unreadCount} unread
                  </Badge>
                  <Button variant='secondary' size='sm' onClick={handleMarkAllRead}>
                    Mark all read
                  </Button>
                </div>
              )}
            </div>
          </div>
        </div>

        <Card>
          <CardHeader className='pb-3'>
            <CardTitle className='text-base'>Notifications</CardTitle>
          </CardHeader>
          <CardContent className='p-0'>
            {loading ? (
              <div className='p-8 text-sm text-muted-foreground text-center'>Loading...</div>
            ) : notifications.length === 0 ? (
              <div className='p-8 text-sm text-muted-foreground text-center'>No notifications yet</div>
            ) : (
              <div className='divide-y'>
                {notifications.map(n => (
                  <div
                    key={n.id}
                    className={`flex items-start gap-4 p-4 ${!n.read ? 'bg-muted/30' : ''}`}
                  >
                    <Badge variant={getTypeVariant(n.type)} className='mt-0.5 shrink-0 capitalize'>
                      {n.type}
                    </Badge>
                    <div className='flex-1 min-w-0'>
                      <p className='text-sm font-medium'>{n.title}</p>
                      <p className='text-sm text-muted-foreground mt-0.5'>{n.message}</p>
                      <p className='text-xs text-muted-foreground/60 mt-1'>{timeAgo(n.created_at)}</p>
                    </div>
                    {!n.read && (
                      <Button
                        variant='ghost'
                        size='sm'
                        className='shrink-0'
                        onClick={() => handleMarkRead(n.id)}
                      >
                        Dismiss
                      </Button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <p className='text-xs text-muted-foreground text-center pt-2'>
          RetinaXAI · Administration
        </p>
      </div>
    </PageContainer>
  );
}
