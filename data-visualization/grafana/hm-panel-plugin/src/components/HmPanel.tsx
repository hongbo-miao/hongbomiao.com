import React, { useState } from 'react';
import { PanelProps } from '@grafana/data';
import { SimpleOptions } from 'types';
import { css, cx } from '@emotion/css';
import { Button, InlineLabel, useStyles2 } from '@grafana/ui';

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

function HmPanel(props: Props) {
  const { options, data, width, height } = props;

  const styles = useStyles2(getStyles);
  const [counter, setCounter] = useState(0);

  const onIncreaseCounter = () => {
    setCounter(previousCounter => previousCounter + 1);
  };

  return (
    <div>
      <InlineLabel width="auto" tooltip="Counter">
        {counter}
      </InlineLabel>
      <Button variant="primary" type="button" onClick={onIncreaseCounter}>
        Increase counter
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

export default HmPanel;
