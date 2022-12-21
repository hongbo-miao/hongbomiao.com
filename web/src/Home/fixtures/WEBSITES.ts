import githubLogo from '../../App/images/github.svg';
import linkedinLogo from '../../App/images/linkedin.svg';
import pinterestLogo from '../../App/images/pinterest.svg';
import twitterLogo from '../../App/images/twitter.svg';
import Website from '../../App/types/Website';

const WEBSITES: ReadonlyArray<Website> = [
  {
    name: 'Pinterest',
    src: pinterestLogo,
    url: 'https://www.pinterest.com/Hongbo_Miao/',
  },
  {
    name: 'GitHub',
    src: githubLogo,
    url: 'https://github.com/Hongbo-Miao',
  },
  {
    name: 'Twitter',
    src: twitterLogo,
    url: 'https://twitter.com/Hongbo_Miao',
  },
  {
    name: 'LinkedIn',
    src: linkedinLogo,
    url: 'https://www.linkedin.com/in/hongbomiao/',
  },
];

export default WEBSITES;
