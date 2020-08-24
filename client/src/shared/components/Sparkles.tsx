import React, { ReactNode } from 'react';
import styled from 'styled-components';
import useRandomInterval from '../hooks/useRandomInterval.hook';
import generateSparkle from '../utils/generateSparkle';
import HmSparkle from './Sparkle';

const Wrapper = styled.span`
  display: inline-block;
  position: relative;
`;
const ChildWrapper = styled.div`
  position: relative;
  z-index: 1;
`;

interface Props {
  children: ReactNode;
}

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
  return (
    <Wrapper>
      {sparkles.map((sparkle) => (
        <HmSparkle key={sparkle.id} color={sparkle.color} size={sparkle.size} style={sparkle.style} />
      ))}
      <ChildWrapper>{children}</ChildWrapper>
    </Wrapper>
  );
};

export default Sparkles;
