@0xa39ce6c189f77dab;

struct Telemetry {
  timestamp @0 :Text;
  entries @1 :List(Entry);
}

struct Entry {
  name @0 :Text;
  data :union {
    missing @1 :Void;
    value @2 :Float64;
  }
}
