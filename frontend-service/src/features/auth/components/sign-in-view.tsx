'use client';

import Image from 'next/image';
import { motion, useReducedMotion } from 'motion/react';
import UserAuthForm from './user-auth-form';
import { fadeIn, slideInUp } from '@/lib/animations';
import { SamayaLogo } from '@/components/auth/samaya-logo';

export default function SignInView() {
  const shouldReduceMotion = useReducedMotion();

  return (
    <div className="relative h-screen flex-col items-center justify-center md:grid lg:max-w-none lg:grid-cols-2 lg:px-0">
      <div className="relative hidden h-full flex-col bg-muted p-10 text-white lg:flex dark:border-r">
        <div className="absolute inset-0">
          <Image
            src="https://images.unsplash.com/photo-1579684385127-1ef15d508118?w=1200&q=80"
            alt="Medical Technology"
            fill
            className="h-full w-full object-cover opacity-30"
            unoptimized
          />
          <div className="absolute inset-0 bg-gradient-to-t from-[#0a2e3e] via-[#0a2e3e]/80 to-transparent" />
        </div>
        <div className="relative z-20 flex items-center text-lg font-medium">
          <Image
            src="https://www.samayahospital.ae/home/images/logo.png"
            alt="Samaya Specialized Center"
            width={32}
            height={32}
            className="mr-3 h-8 w-auto"
            unoptimized
          />
          <span className="text-[var(--brand-teal)]">RetinaXAI</span>
        </div>
        <div className="relative z-20 mt-auto">
          <blockquote className="space-y-2">
            <p className="text-lg">
              &ldquo;AI-assisted retinal grading with explainable predictions
              for every diagnosis.&rdquo;
            </p>
            <footer className="text-sm text-[var(--brand-teal)]/80">
              Clinical Decision Support
            </footer>
          </blockquote>
        </div>
      </div>
      <motion.div
        variants={shouldReduceMotion ? {} : fadeIn}
        initial="hidden"
        animate="visible"
        className="flex h-full items-center p-4 lg:p-8"
      >
        <div className="mx-auto flex w-full flex-col justify-center space-y-6 sm:w-[350px]">
          <motion.div
            variants={shouldReduceMotion ? {} : slideInUp}
            initial="hidden"
            animate="visible"
            className="flex justify-center"
          >
            <SamayaLogo size="lg" animate={!shouldReduceMotion} variant="full" />
          </motion.div>
          <div className="flex flex-col space-y-2 text-center">
            <h1 className="text-2xl font-semibold tracking-tight">
              Sign in to your account
            </h1>
            <p className="text-sm text-muted-foreground">
              Enter your credentials to access the dashboard
            </p>
          </div>
          <UserAuthForm />
        </div>
      </motion.div>
    </div>
  );
}