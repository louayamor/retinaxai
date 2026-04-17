'use client';

import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from '@/hooks/use-websocket';
import { fetchNotifications, markNotificationsRead, markAllNotificationsRead, clearNotifications, type NotificationItem } from '@/lib/api';
import { Bell, X, Check, Trash2, Loader2, Brain, FlaskConical, Sparkles, BookOpen, FileText, AlertCircle, AlertTriangle, Mail } from 'lucide-react';

const STORAGE_KEY = 'rxa_notifications';

interface Notification extends NotificationItem {
  timestamp: string;
}

function getNotificationTypeIcon(type: string) {
  switch (type) {
    case 'training':
      return <Brain className="h-5 w-5 text-[var(--brand-teal)]" />;
    case 'prediction':
      return <FlaskConical className="h-5 w-5 text-[var(--brand-teal)]" />;
    case 'xai':
      return <Sparkles className="h-5 w-5 text-purple-500" />;
    case 'rag':
      return <BookOpen className="h-5 w-5 text-blue-500" />;
    case 'report':
      return <FileText className="h-5 w-5 text-[var(--brand-gold)]" />;
    case 'error':
    case 'training_error':
      return <AlertCircle className="h-5 w-5 text-red-500" />;
    case 'warning':
      return <AlertTriangle className="h-5 w-5 text-amber-500" />;
    default:
      return <Mail className="h-5 w-5 text-gray-500" />;
  }
}

export function NotificationCenter() {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { connected, subscribe } = useWebSocket();

  // Load notifications from API on mount
  const loadNotifications = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchNotifications(false, 50, 0);
      // Convert API format to local format
      const converted: Notification[] = data.map(n => ({
        ...n,
        timestamp: n.created_at,
      }));
      setNotifications(converted);
      // Also persist to localStorage as backup
      localStorage.setItem(STORAGE_KEY, JSON.stringify(converted));
    } catch (err) {
      console.error('Failed to load notifications:', err);
      // Fallback to localStorage
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        setNotifications(JSON.parse(stored));
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadNotifications();
  }, [loadNotifications]);

  // Subscribe to real-time WebSocket notifications
  useEffect(() => {
    const unsubscribe = subscribe('notification', (data: unknown) => {
      const wsNotif = data as Notification;
      const newNotif: Notification = {
        id: wsNotif.id || crypto.randomUUID(),
        type: wsNotif.type || 'general',
        title: wsNotif.title || '',
        message: wsNotif.message || '',
        read: false,
        created_at: wsNotif.timestamp || new Date().toISOString(),
        timestamp: wsNotif.timestamp || new Date().toISOString(),
        user_id: wsNotif.user_id || null,
      };
      setNotifications(prev => [newNotif, ...prev].slice(0, 50));
      // Update localStorage
      const updated = [newNotif, ...notifications].slice(0, 50);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    });

    return unsubscribe;
  }, [subscribe, notifications]);

  const unreadCount = notifications.filter(n => !n.read).length;

  const handleMarkAsRead = async (id: string) => {
    // Optimistic update
    setNotifications(prev => prev.map(n => n.id === id ? { ...n, read: true } : n));
    try {
      await markNotificationsRead([id]);
    } catch (err) {
      console.error('Failed to mark as read:', err);
    }
  };

  const handleMarkAllRead = async () => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })));
    try {
      await markAllNotificationsRead();
    } catch (err) {
      console.error('Failed to mark all as read:', err);
    }
  };

  const handleClearAll = async () => {
    setNotifications([]);
    try {
      await clearNotifications(true); // Clear only read notifications
      localStorage.removeItem(STORAGE_KEY);
    } catch (err) {
      console.error('Failed to clear notifications:', err);
    }
  };

  const handleClickNotif = async (notif: Notification) => {
    if (!notif.read) {
      await handleMarkAsRead(notif.id);
    }
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="relative p-2 rounded-lg hover:bg-sidebar-accent transition-colors"
        aria-label="Notifications"
      >
        <Bell className="w-5 h-5 text-sidebar-foreground" />
        {unreadCount > 0 && (
          <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs w-5 h-5 rounded-full flex items-center justify-center font-medium">
            {unreadCount > 9 ? '9+' : unreadCount}
          </span>
        )}
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-96 bg-sidebar text-sidebar-foreground rounded-lg shadow-xl border border-sidebar-border z-50 overflow-hidden">
          <div className="p-3 border-b border-sidebar-border flex justify-between items-center bg-sidebar-accent">
            <h3 className="font-semibold text-sidebar-foreground flex items-center gap-2">
              <Bell className="w-4 h-4" />
              Notifications
            </h3>
            <div className="flex items-center gap-1">
              {unreadCount > 0 && (
                <button
                  onClick={handleMarkAllRead}
                  className="p-1.5 rounded hover:bg-sidebar/80 text-sidebar-foreground"
                  title="Mark all as read"
                >
                  <Check className="w-4 h-4" />
                </button>
              )}
              {notifications.length > 0 && (
                <button
                  onClick={handleClearAll}
                  className="p-1.5 rounded hover:bg-sidebar/80 text-sidebar-foreground"
                  title="Clear all"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
              <button
                onClick={() => setIsOpen(false)}
                className="p-1.5 rounded hover:bg-sidebar/80 text-sidebar-foreground ml-1"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="max-h-96 overflow-y-auto">
            {loading ? (
              <div className="p-8 text-center text-sidebar-foreground/70 flex items-center justify-center gap-2">
                <Loader2 className="w-5 h-5 animate-spin" />
                Loading...
              </div>
            ) : notifications.length === 0 ? (
              <div className="p-8 text-center text-sidebar-foreground/70">
                No notifications yet
              </div>
            ) : (
              notifications.map(notif => (
                <div
                  key={notif.id}
                  onClick={() => handleClickNotif(notif)}
                  className={`p-3 border-b border-sidebar-border/50 cursor-pointer hover:bg-sidebar-accent transition-colors ${
                    !notif.read ? 'bg-sidebar-accent/30' : ''
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <span className="text-lg" title={notif.type}>
                      {getNotificationTypeIcon(notif.type)}
                    </span>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-sidebar-foreground truncate">
                        {notif.title || notif.message.split('.')[0]}
                      </p>
                      <p className="text-sm text-sidebar-foreground/70 line-clamp-2">
                        {notif.message}
                      </p>
                      <p className="text-xs text-sidebar-foreground/50 mt-1">
                        {new Date(notif.timestamp).toLocaleString()}
                      </p>
                    </div>
                    {!notif.read && (
                      <div className="w-2 h-2 rounded-full bg-[var(--brand-teal)] flex-shrink-0 mt-2" />
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
          
          {notifications.length > 0 && (
            <div className="p-2 border-t border-sidebar-border bg-sidebar-accent text-center">
              <button 
                onClick={loadNotifications}
                className="text-xs text-sidebar-foreground/70 hover:text-sidebar-foreground"
              >
                Refresh
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}