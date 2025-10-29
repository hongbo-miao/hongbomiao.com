import { createFileRoute } from '@tanstack/react-router';
import Fire from '@/Fire/components/Fire';

export const Route = createFileRoute('/fire')({
  component: Fire,
});
