@0xe3a426facb631d59;

struct Transcription {
  streamId @0 :Text;
  timestampNs @1 :Int64;
  text @2 :Text;
  language @3 :Text;
  durationS @4 :Float64;
  segmentStartS @5 :Float64;
  segmentEndS @6 :Float64;
  words @7 :List(Word);
}

struct Word {
  word @0 :Text;
  startS @1 :Float64;
  endS @2 :Float64;
  probability @3 :Float64;
}
