import React, { ReactNode } from 'react';
import useRandomInterval from '../hooks/useRandomInterval.hook';
import generateSparkle from '../utils/generateSparkle';
import HmSparkle from './Sparkle';
import styles from './Sparkles.module.css';

type Props = {
  children: ReactNode;
};

const Sparkles: React.FC<Props> = (props) => {
  const { children } = props;

  const [sparkles, setSparkles] = React.useState(() => {
    return Array(3).map(() => generateSparkle());
  });

  useRandomInterval(
    () => {
      const sparkle = generateSparkle();
      const now = Date.now();
      setSparkles([...sparkles.filter((sp) => now - sp.createdAt < 750), sparkle]);
    },
    50,
    450
  );

  const sparkleItems = sparkles.map((sparkle) => (
    <HmSparkle key={sparkle.id} color={sparkle.color} size={sparkle.size} style={sparkle.style} />
  ));

  return (
    <span className={styles.hmWrapper}>
      {sparkleItems}
      <div className={styles.hmChildrenWrapper}>{children}</div>
    </span>
  );
};

export default Sparkles;
