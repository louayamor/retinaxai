'use client';

import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage
} from '@/components/ui/form';
import { loginUser } from '@/lib/api';
import { useAuth } from '@/providers/auth-context';
import type { UserRole } from '@/lib/auth';
const loginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(1, 'Password is required'),
});

type LoginValues = z.infer<typeof loginSchema>;

export default function UserAuthForm() {
  const { setUser } = useAuth();
  const [serverError, setServerError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const form = useForm<LoginValues>({
    resolver: zodResolver(loginSchema),
    defaultValues: { email: '', password: '' }
  });

  async function onLogin(values: LoginValues) {
    setLoading(true);
    setServerError(null);
    try {
      await loginUser(values);
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/v1/auth/me`,
        { credentials: 'include' }
      );
      if (res.ok) {
        const data = await res.json();
        setUser({
          id: data.id,
          username: data.username,
          email: data.email,
          role: data.role as UserRole,
        });
      } else {
        setServerError('Failed to load user profile.');
      }
    } catch (err: unknown) {
      const e = err as { message?: string; status?: number };
      setServerError(
        e?.status === 401
          ? 'Invalid username or password.'
          : (e?.message ?? 'Login failed.')
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <Form form={form} onSubmit={form.handleSubmit(onLogin)} className='space-y-4'>
      {serverError && (
        <p className='text-sm text-destructive'>{serverError}</p>
      )}
      <FormField
        control={form.control}
        name='email'
        render={({ field }) => (
          <FormItem>
            <FormLabel>Email</FormLabel>
            <FormControl>
              <Input type='email' placeholder='doctor@hospital.com' {...field} />
            </FormControl>
            <FormMessage />
          </FormItem>
        )}
      />
      <FormField
        control={form.control}
        name='password'
        render={({ field }) => (
          <FormItem>
            <FormLabel>Password</FormLabel>
            <FormControl>
              <Input type='password' placeholder='••••••••' {...field} />
            </FormControl>
            <FormMessage />
          </FormItem>
        )}
      />
      <Button type='submit' className='w-full' disabled={loading}>
        {loading ? 'Signing in...' : 'Sign In'}
      </Button>
    </Form>
  );
}
