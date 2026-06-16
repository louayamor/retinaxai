'use client';

import { useIdleTimeout } from '@/hooks/use-idle-timeout';
import { useAuth } from '@/providers/auth-context';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Button } from '@/components/ui/button';

const TIMEOUT_MS = 15 * 60 * 1000;
const WARNING_MS = 60 * 1000;

export function SessionTimeoutDialog() {
  const { user, logout } = useAuth();
  const { showWarning, resetTimer } = useIdleTimeout({
    timeoutMs: TIMEOUT_MS,
    warningMs: WARNING_MS,
    onTimeout: () => { void logout(); },
    enabled: !!user,
  });

  if (!showWarning) return null;

  return (
    <AlertDialog open>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Session expiring soon</AlertDialogTitle>
          <AlertDialogDescription>
            Your session will expire in 1 minute due to inactivity. Press anywhere to stay signed in.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <Button onClick={resetTimer} variant='default'>
            Stay signed in
          </Button>
          <AlertDialogAction onClick={() => { void logout(); }}>
            Sign out
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
