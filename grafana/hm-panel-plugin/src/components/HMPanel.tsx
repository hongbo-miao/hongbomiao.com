import React from 'react';
import { PanelProps } from '@grafana/data';
import { SimpleOptions } from 'types';
import { css, cx } from '@emotion/css';
import { Button, InlineLabel, useStyles2 } from '@grafana/ui';
import useSeed from '../hooks/useSeed';
import { QueryClientProvider } from '@tanstack/react-query';
import queryClient from '../utils/queryClient';

interface Props extends PanelProps<SimpleOptions> {}

const getStyles = () => {
  return {
    wrapper: css`
      font-family: Open Sans;
      position: relative;
    `,
    textBox: css`
      position: absolute;
      bottom: 0;
      left: 0;
      padding: 10px;
    `,
  };
};

function HMPanel(props: Props) {
  const { options, data, width, height } = props;

  const styles = useStyles2(getStyles);
  const { seed } = useSeed();

  const onIncreaseSeed = () => {
    console.log('Hi');
  };

  return (
    <div>
      <InlineLabel width="auto" tooltip="Seed Number">
        {seed?.seedNumber}
      </InlineLabel>
      <Button variant="primary" type="button" onClick={onIncreaseSeed}>
        Click seed
      </Button>
      <div
        className={cx(
          styles.wrapper,
          css`
            width: ${width}px;
            height: ${height}px;
          `
        )}
      >
        <div className={styles.textBox}>
          {options.showSeriesCount && <div>Number of series: {data.series.length}</div>}
          <div>Text option value: {options.text}</div>
        </div>
      </div>
    </div>
  );
}

function addQueryClientProvider(Component: React.FC<Props>) {
  // eslint-disable-next-line react/display-name
  return (props: Props) => (
    <QueryClientProvider client={queryClient}>
      <Component {...props} />
    </QueryClientProvider>
  );
}

export default addQueryClientProvider(HMPanel);
