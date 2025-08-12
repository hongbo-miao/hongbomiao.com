import React, { ReactNode } from 'react';
import useRandomInterval from '../hooks/useRandomInterval.hook';
import generateSparkle from '../utils/generateSparkle';
import HmSparkle from './Sparkle';
import './Sparkles.css';

type Props = {
  children: ReactNode;
};

function Sparkles(props: Props) {
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
    450,
  );

  const sparkleItems = sparkles.map((sparkle) => (
    <HmSparkle key={sparkle.id} color={sparkle.color} size={sparkle.size} style={sparkle.style} />
  ));

  return (
    <span className="hm-sparkles-wrapper">
      {sparkleItems}
      <div className="hm-sparkles-children-wrapper">{children}</div>
    </span>
  );
}

export default Sparkles;
