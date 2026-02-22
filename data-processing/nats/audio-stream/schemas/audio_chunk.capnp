@0xb7c5f36c8e9a2d01;

struct AudioChunk {
  timestampNs @0 :Int64;
  sampleRateHz @1 :UInt32;
  audioData @2 :Data;
  audioFormat @3 :Text;
}
