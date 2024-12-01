import type { Meta, StoryObj } from '@storybook/react';
import HmCopyright from './Copyright';

type Story = StoryObj<typeof HmCopyright>;

export const Primary: Story = {
  args: {
    year: 1990,
  },
};

const meta: Meta<typeof HmCopyright> = {
  component: HmCopyright,
  title: 'Copyright',
};

export default meta;
