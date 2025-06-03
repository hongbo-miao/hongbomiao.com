#include <core.p4>
#include <v1model.p4>

const int MAX_HOPS = 10;
const int STANDARD = 0x00;
const int HOPS = 0x01;

typedef standard_metadata_t std_meta_t;

header type_t {
  bit<8> tag;
}

header hop_t {
  bit<8> port;
  bit<8> bos;
}

header standard_t {
  bit<8> src;
  bit<8> dst;
}

struct headers_t {
  type_t type;
  hop_t[MAX_HOPS] hops;
  standard_t standard;
}

struct meta_t {}

parser MyParser(packet_in pkt, out headers_t hdr, inout meta_t meta, inout std_meta_t std_meta) {
  state start {
    pkt.extract(hdr.type);
    transition select(hdr.type.tag) {
      HOPS: parse_hops;
      STANDARD: parse_standard;
      default: accept;
    }
  }
  state parse_hops {
    pkt.extract(hdr.hops.next);
    transition select(hdr.hops.last.bos) {
      1: parse_standard;
      default: parse_hops;
    }
  }

  state parse_standard {
    pkt.extract(hdr.standard);
    transition accept;
  }
}

control MyVerifyChecksum(inout headers_t hdr, inout meta_t meta) {
  apply {}
}

control MyComputeChecksum(inout headers_t hdr, inout meta_t meta) {
  apply {}
}

control MyIngress(inout headers_t hdr, inout meta_t meta, inout std_meta_t std_meta) {
  action allow() { }
  action deny() { std_meta.egress_spec = 9w511; }

  table acl {
    key = {
      hdr.standard.src : exact;
      hdr.standard.dst : exact;
    }
    actions = {
      allow;
      deny;
    }
    const entries = {
      (0xCC, 0xDD) : deny();
    }
    default_action = allow();
  }
  apply {
    std_meta.egress_spec = (bit<9>) hdr.hops[0].port;
    hdr.hops.pop_front(1);
    if (!hdr.hops[0].isValid()) {
      hdr.type.tag = 0x00;
    }
    acl.apply();
  }
}

control MyEgress(inout headers_t hdr, inout meta_t meta, inout std_meta_t std_meta) {
  apply {}
}

control MyDeparser(packet_out pkt, in headers_t hdr) {
  apply {
    pkt.emit(hdr.type);
    pkt.emit(hdr.hops);
    pkt.emit(hdr.standard);
  }
}

V1Switch(MyParser(), MyVerifyChecksum(), MyIngress(), MyEgress(), MyComputeChecksum(), MyDeparser()) main;
