import { createFileRoute } from '@tanstack/react-router';
import Emergency from '@/Emergency/components/Emergency';

export const Route = createFileRoute('/emergency')({
  component: Emergency,
});
