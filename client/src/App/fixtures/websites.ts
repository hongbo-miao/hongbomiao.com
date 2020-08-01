import Website from '../typings/website';
import githubLogo from '../images/github.svg';
import linkedinLogo from '../images/linkedin.svg';
import pinterestLogo from '../images/pinterest.svg';
import twitterLogo from '../images/twitter.svg';

const WEBSITES: Website[] = [
  {
    name: 'Pinterest',
    src: pinterestLogo,
    url: 'https://www.pinterest.com/hongbo_miao/',
  },
  {
    name: 'GitHub',
    src: githubLogo,
    url: 'https://github.com/hongbo-miao/',
  },
  {
    name: 'Twitter',
    src: twitterLogo,
    url: 'https://twitter.com/hongbo_miao/',
  },
  {
    name: 'LinkedIn',
    src: linkedinLogo,
    url: 'https://www.linkedin.com/in/hongbomiao/',
  },
];

export default WEBSITES;
