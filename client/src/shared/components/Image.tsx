import React from 'react';
import styles from './Image.module.css';

interface Props {
  webpSrc: string;
  fallbackSrc: string;
  height: string;
  width: string;
  alt: string;
}

const Image: React.FC<Props> = (props: Props) => {
  const { webpSrc, fallbackSrc, height, width, alt } = props;
  return (
    <picture className={styles.hmPicture}>
      <source type="image/webp" srcSet={webpSrc} />
      <img src={fallbackSrc} height={height} width={width} alt={alt} />
    </picture>
  );
};

export default Image;
