import type { Meta, StoryObj } from '@storybook/react';
import WEBSITES from '../../Home/fixtures/WEBSITES';
import HmSocialList from './SocialList';

type Story = StoryObj<typeof HmSocialList>;

export const Primary: Story = {
  args: {
    websites: WEBSITES,
  },
};

export const Empty: Story = {
  args: {
    websites: [],
  },
};

const meta: Meta<typeof HmSocialList> = {
  component: HmSocialList,
  title: 'SocialList',
};

export default meta;
