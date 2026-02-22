@0xe3a426facb631d59;

struct Transcription {
  streamId @0 :Text;
  timestampNs @1 :Int64;
  text @2 :Text;
  language @3 :Text;
  durationS @4 :Float64;
  words @5 :List(Word);
  sampleRateHz @6 :UInt32;
  audioData @7 :Data;
  audioFormat @8 :Text;
}

struct Word {
  word @0 :Text;
  startS @1 :Float64;
  endS @2 :Float64;
  probability @3 :Float64;
}
