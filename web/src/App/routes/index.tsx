import { createFileRoute } from '@tanstack/react-router';
import HmHome from '../../Home/components/Home';

export const Route = createFileRoute('/')({
  component: HmHome,
});
