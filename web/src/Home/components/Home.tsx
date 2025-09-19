import React from 'react';
import springWaltzMp3 from '@/Home/audio/spring-waltz.mp3';
import HmFooter from '@/Home/components/Footer';
import { ThemeToggleButton } from '@/Home/components/ThemeToggleButton';
import hatAvif from '@/Home/images/hat.avif';
import hatPng from '@/Home/images/hat.png';
import magicAvif from '@/Home/images/magic.avif';
import magicPng from '@/Home/images/magic.png';
import config from '@/config';
import useHealthSubscription from '@/health/hooks/useHealthSubscription';
import HmAudioPlayer from '@/shared/components/AudioPlayer';
import HmImage from '@/shared/components/Image';
import HmSparkles from '@/shared/components/Sparkles';
import analytics from '@/shared/utils/analytics';
import '@/Home/components/Home.css';

function Home() {
  useHealthSubscription();

  React.useEffect(() => {
    analytics.page();
  }, []);

  // Static content instead of auth-based data
  const name = 'Hongbo Miao';
  const bio = 'Making magic happen';

  return (
    <div className="hm-home">
      <ThemeToggleButton />
      <div className="hm-body">
        <div className="mx-auto hm-home-container">
          <div className="hm-name-container">
            <h1 className="hm-name">{name}</h1>
            <div className="hm-audio-player-wrapper">
              <HmAudioPlayer audioSrc={springWaltzMp3} />
            </div>
          </div>
          <HmSparkles>
            <a className="hm-bio-container" href={config.githubUrl} target="_blank" rel="noopener noreferrer">
              <HmImage
                avifSrc={hatAvif}
                fallbackSrc={hatPng}
                style={{ height: '26px', width: '26px' }}
                alt="Magical Hat"
              />
              <div className="hm-bio">{bio}</div>
              <HmImage
                avifSrc={magicAvif}
                fallbackSrc={magicPng}
                style={{ height: '26px', width: '26px' }}
                alt="Magic"
              />
            </a>
          </HmSparkles>
        </div>
      </div>
      <HmFooter />
    </div>
  );
}

export default Home;
