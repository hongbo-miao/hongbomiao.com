import React from 'react';

import styles from './Image.module.css';

interface Props {
  alt: string;
  className?: string;
  src: string;
  webpSrc: string;
}

const Image: React.FC<Props> = (props: Props) => {
  const { alt, className, src, webpSrc } = props;
  return (
    <picture className={styles.hmPicture}>
      <source type="image/webp" srcSet={webpSrc} />
      <img className={className} src={src} alt={alt} />
    </picture>
  );
};

export default Image;
