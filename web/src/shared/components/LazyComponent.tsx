import React, { ReactNode, Suspense } from 'react';
import HmLoading from './Loading';

type Props = {
  children: ReactNode;
};

function LazyComponent(props: Props) {
  const { children } = props;
  return <Suspense fallback={<HmLoading />}>{children}</Suspense>;
}

export default LazyComponent;
