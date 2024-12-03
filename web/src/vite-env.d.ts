/// <reference types="vite/client" />

declare module '*.css' {
  const classes: { [key: string]: string };
  export default classes;
}

declare module '*.avif' {
  const src: string;
  export default src;
}

declare module '*.png' {
  const src: string;
  export default src;
}

declare module '*.svg' {
  const src: string;
  export default src;
}

declare module '*.gif' {
  const src: string;
  export default src;
}

declare module '*.mp3' {
  const src: string;
  export default src;
}
