import React, { ReactNode, Suspense } from 'react';

import HmLoading from './Loading';

interface Props {
  children: ReactNode;
}

const LazyComponent: React.FC<Props> = (props: Props) => {
  const { children } = props;
  return <Suspense fallback={<HmLoading />}>{children}</Suspense>;
};

export default LazyComponent;
