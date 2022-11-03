import React from 'react';
import { PanelProps } from '@grafana/data';
import { SimpleOptions } from 'types';
import { css, cx } from '@emotion/css';
import { Button, useStyles2, useTheme2 } from '@grafana/ui';

interface Props extends PanelProps<SimpleOptions> {}

const getStyles = () => {
  return {
    wrapper: css`
      font-family: Open Sans;
      position: relative;
    `,
    svg: css`
      position: absolute;
      top: 0;
      left: 0;
    `,
    textBox: css`
      position: absolute;
      bottom: 0;
      left: 0;
      padding: 10px;
    `,
  };
};

const SimplePanel: React.FC<Props> = ({ options, data, width, height }) => {
  const theme = useTheme2();
  const styles = useStyles2(getStyles);

  console.log('options', options);
  console.log('data', data);
  console.log('height', height);
  console.log('width', width);

  const onClick = () => {
    console.log('Hi');
  };

  return (
    <div>
      <Button variant="primary" type="button" onClick={onClick}>
        Click me
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
        <svg
          className={styles.svg}
          width={width}
          height={height}
          xmlns="http://www.w3.org/2000/svg"
          xmlnsXlink="http://www.w3.org/1999/xlink"
          viewBox={`-${width / 2} -${height / 2} ${width} ${height}`}
        >
          <g>
            <circle style={{ fill: theme.colors.primary.main }} r={100} />
          </g>
        </svg>

        <div className={styles.textBox}>
          {options.showSeriesCount && <div>Number of series: {data.series.length}</div>}
          <div>Text option value: {options.text}</div>
        </div>
      </div>
    </div>
  );
};

export default SimplePanel;
