import { useAudio } from 'react-use';
import flashingMusicalNoteGIF from '@/shared/images/musical-note-flashing.gif';
import playingMusicalNoteGIF from '@/shared/images/musical-note-playing.gif';
import '@/shared/components/AudioPlayer.css';

interface Props {
  audioSrc: string;
}

function AudioPlayer(props: Props) {
  const { audioSrc } = props;
  const [audioElement, audioState, audioControls] = useAudio({
    src: audioSrc,
  });

  const onTogglePlay = () => {
    if (audioState.paused) {
      audioControls.play();
    } else {
      audioControls.pause();
    }
  };

  const musicalNoteGIF = audioState.paused ? flashingMusicalNoteGIF : playingMusicalNoteGIF;

  return (
    <>
      {audioElement}
      <button className="button hm-play-button" type="button" onClick={onTogglePlay}>
        <img className="hm-musical-note" width="20" height="20" src={musicalNoteGIF} alt="Music" />
      </button>
    </>
  );
}

export default AudioPlayer;
