import React from 'react';
import './Image.css';

type Props = {
  avifSrc: string;
  fallbackSrc: string;
  alt: string;
  style: {
    height: string;
    width: string;
  };
};

function Image(props: Props) {
  const { avifSrc, fallbackSrc, style, alt } = props;
  return (
    <picture className="hm-picture">
      <source type="image/avif" srcSet={avifSrc} />
      <img src={fallbackSrc} style={style} alt={alt} />
    </picture>
  );
}

export default Image;
