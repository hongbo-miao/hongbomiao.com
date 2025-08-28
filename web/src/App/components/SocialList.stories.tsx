import type { Meta, StoryObj } from '@storybook/react';
import HmSocialList from '@/App/components/SocialList';
import WEBSITES from '@/Home/fixtures/WEBSITES';

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
