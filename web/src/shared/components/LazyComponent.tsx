import { Suspense, type ReactNode } from 'react';
import HmLoading from '@/shared/components/Loading';

type Props = {
  children: ReactNode;
};

function LazyComponent(props: Props) {
  const { children } = props;
  return <Suspense fallback={<HmLoading />}>{children}</Suspense>;
}

export default LazyComponent;
