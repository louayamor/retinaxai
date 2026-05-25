'use client';

import { useEffect, useState } from 'react';
import { getAdminUsers, updateAdminUser } from '@/lib/api';
import type { AdminUserItem } from '@/lib/api';
import PageContainer from '@/components/layout/page-container';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';

const ROLE_BADGE: Record<string, 'info' | 'warning' | 'default'> = {
  doctor: 'info',
  engineer: 'warning',
  admin: 'default',
};

export default function AdminUsersPage() {
  const [users, setUsers] = useState<AdminUserItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingUser, setEditingUser] = useState<AdminUserItem | null>(null);
  const [editRole, setEditRole] = useState('');

  useEffect(() => { fetchUsers(); }, []);

  async function fetchUsers() {
    setLoading(true);
    try {
      const res = await getAdminUsers();
      setUsers(res.items);
    } catch { /* ignore */ }
    setLoading(false);
  }

  async function handleSaveRole() {
    if (!editingUser) return;
    try {
      await updateAdminUser(editingUser.id, { role: editRole });
      setUsers(prev => prev.map(u => u.id === editingUser.id ? { ...u, role: editRole } : u));
      setEditingUser(null);
    } catch { /* ignore */ }
  }

  async function handleToggleActive(user: AdminUserItem) {
    try {
      await updateAdminUser(user.id, { is_active: !user.is_active });
      setUsers(prev => prev.map(u => u.id === user.id ? { ...u, is_active: !u.is_active } : u));
    } catch { /* ignore */ }
  }

  return (
    <PageContainer>
      <div className='flex flex-1 flex-col gap-6 min-h-0'>
        <div className='rounded-lg border bg-gradient-to-r from-slate-900 via-cyan-900 to-teal-900 p-5 text-white shadow-sm'>
          <div className='relative z-10 space-y-1'>
            <h1 className='text-xl font-bold tracking-tight'>User Management</h1>
            <p className='max-w-xl text-sm text-white/70'>
              Manage platform users, roles, and account status
            </p>
          </div>
        </div>

        <div className='rounded-lg border'>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Username</TableHead>
                <TableHead>Email</TableHead>
                <TableHead>Role</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Created</TableHead>
                <TableHead className='text-right'>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                <TableRow>
                  <TableCell colSpan={6} className='text-center text-muted-foreground py-8'>Loading...</TableCell>
                </TableRow>
              ) : users.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className='text-center text-muted-foreground py-8'>No users found</TableCell>
                </TableRow>
              ) : (
                users.map(user => (
                  <TableRow key={user.id}>
                    <TableCell className='font-medium'>{user.username}</TableCell>
                    <TableCell>{user.email}</TableCell>
                    <TableCell>
                      <Badge variant={ROLE_BADGE[user.role] ?? 'outline'} className='capitalize'>
                        {user.role}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant={user.is_active ? 'success' : 'destructive'}>
                        {user.is_active ? 'Active' : 'Inactive'}
                      </Badge>
                    </TableCell>
                    <TableCell className='text-muted-foreground text-sm'>
                      {new Date(user.created_at).toLocaleDateString()}
                    </TableCell>
                    <TableCell className='text-right'>
                      <div className='flex justify-end gap-2'>
                        <Button
                          variant='outline'
                          size='sm'
                          onClick={() => { setEditingUser(user); setEditRole(user.role); }}
                        >
                          Edit Role
                        </Button>
                        <Button
                          variant={user.is_active ? 'destructive' : 'default'}
                          size='sm'
                          onClick={() => handleToggleActive(user)}
                        >
                          {user.is_active ? 'Deactivate' : 'Activate'}
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </div>

      <Dialog open={!!editingUser} onOpenChange={(open) => { if (!open) setEditingUser(null); }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Role — {editingUser?.username}</DialogTitle>
          </DialogHeader>
          <Select value={editRole} onValueChange={setEditRole}>
            <SelectTrigger>
              <SelectValue placeholder='Select role' />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value='doctor'>Doctor</SelectItem>
              <SelectItem value='engineer'>Engineer</SelectItem>
              <SelectItem value='admin'>Admin</SelectItem>
            </SelectContent>
          </Select>
          <DialogFooter>
            <Button variant='outline' onClick={() => setEditingUser(null)}>Cancel</Button>
            <Button onClick={handleSaveRole}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </PageContainer>
  );
}
