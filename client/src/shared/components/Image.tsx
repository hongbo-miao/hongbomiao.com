import React from 'react';
import styles from './Image.module.css';

interface Props {
  webpSrc: string;
  fallbackSrc: string;
  alt: string;
  style: {
    height: string;
    width: string;
  };
}

const Image: React.FC<Props> = (props) => {
  const { webpSrc, fallbackSrc, style, alt } = props;
  return (
    <picture className={styles.hmPicture}>
      <source type="image/webp" srcSet={webpSrc} />
      <img src={fallbackSrc} style={style} alt={alt} />
    </picture>
  );
};

export default Image;
