import clsx from 'clsx';
import React from 'react';
import { useAudio } from 'react-use';
import flashingMusicalNoteGIF from '../images/musical-note-flashing.gif';
import playingMusicalNoteGIF from '../images/musical-note-playing.gif';
import styles from './AudioPlayer.module.css';

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
  const playButtonClassName = clsx('button', styles.hmPlayButton);

  return (
    <>
      {audioElement}
      <button className={playButtonClassName} type="button" onClick={onTogglePlay}>
        <img className={styles.hmMusicalNote} width="20" height="20" src={musicalNoteGIF} alt="Music" />
      </button>
    </>
  );
}

export default AudioPlayer;
