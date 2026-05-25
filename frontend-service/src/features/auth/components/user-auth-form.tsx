'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
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
import { loginUser, registerUser } from '@/lib/api';

const loginSchema = z.object({
  email: z.string().email('Enter a valid email'),
  password: z.string().min(1, 'Password is required')
});

const registerSchema = z.object({
  username: z.string().min(3, 'At least 3 characters'),
  email: z.string().email('Enter a valid email'),
  password: z.string().min(8, 'At least 8 characters'),
  confirmPassword: z.string().min(1, 'Please confirm your password'),
}).refine(data => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

type LoginValues = z.infer<typeof loginSchema>;
type RegisterValues = z.infer<typeof registerSchema>;

interface UserAuthFormProps {
  mode: 'login' | 'register';
}

export default function UserAuthForm({ mode }: UserAuthFormProps) {
  const router = useRouter();
  const [serverError, setServerError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const loginForm = useForm<LoginValues>({
    resolver: zodResolver(loginSchema),
    defaultValues: { email: '', password: '' }
  });

  const registerForm = useForm<RegisterValues>({
    resolver: zodResolver(registerSchema),
    defaultValues: { username: '', email: '', password: '', confirmPassword: '' }
  });

  async function onLogin(values: LoginValues) {
    setLoading(true);
    setServerError(null);
    try {
      await loginUser(values);
      window.location.href = '/dashboard/overview';
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

  async function onRegister(values: RegisterValues) {
    setLoading(true);
    setServerError(null);
    try {
      await registerUser(values);
      await loginUser({ email: values.email, password: values.password });
      window.location.href = '/dashboard/overview';
    } catch (err: unknown) {
      const e = err as { message?: string; status?: number };
      setServerError(
        e?.status === 409
          ? 'Username or email already exists.'
          : (e?.message ?? 'Registration failed.')
      );
    } finally {
      setLoading(false);
    }
  }

  if (mode === 'login') {
    return (
      <Form
        form={loginForm}
        onSubmit={loginForm.handleSubmit(onLogin)}
        className='space-y-4'
      >
        {serverError && (
          <p className='text-sm text-destructive'>{serverError}</p>
        )}
        <FormField
          control={loginForm.control}
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
          control={loginForm.control}
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

  return (
    <Form
      form={registerForm}
      onSubmit={registerForm.handleSubmit(onRegister)}
      className='space-y-4'
    >
      {serverError && (
        <p className='text-sm text-destructive'>{serverError}</p>
      )}
      <FormField
        control={registerForm.control}
        name='username'
        render={({ field }) => (
          <FormItem>
            <FormLabel>Username</FormLabel>
            <FormControl>
              <Input placeholder='dr.username' {...field} />
            </FormControl>
            <FormMessage />
          </FormItem>
        )}
      />
      <FormField
        control={registerForm.control}
        name='email'
        render={({ field }) => (
          <FormItem>
            <FormLabel>Email</FormLabel>
            <FormControl>
              <Input type='email' placeholder='doctor@hospital.ae' {...field} />
            </FormControl>
            <FormMessage />
          </FormItem>
        )}
      />
      <FormField
        control={registerForm.control}
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
      <FormField
        control={registerForm.control}
        name='confirmPassword'
        render={({ field }) => (
          <FormItem>
            <FormLabel>Confirm Password</FormLabel>
            <FormControl>
              <Input type='password' placeholder='••••••••' {...field} />
            </FormControl>
            <FormMessage />
          </FormItem>
        )}
      />
      <Button type='submit' className='w-full' disabled={loading}>
        {loading ? 'Creating account...' : 'Create Account'}
      </Button>
    </Form>
  );
}
