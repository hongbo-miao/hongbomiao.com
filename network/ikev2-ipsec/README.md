# IKEv2/IPsec

## What are IKEv2 and IPsec?

These are two separate protocols that work as a pair -- one negotiates, the other protects traffic.

- **IPsec** (Internet Protocol Security) is the suite that actually encrypts and authenticates IP
  traffic. Its data-plane protocol here is **ESP** (Encapsulating Security Payload): take a packet,
  encrypt its payload, wrap it in an ESP header/trailer. IPsec on its own says nothing about *how* the
  two sides agree on keys -- it just uses whatever keys and algorithm it's handed.
- **IKEv2** (Internet Key Exchange, version 2) is the protocol that supplies those keys. It's the
  handshake: the two gateways authenticate each other (here, with a pre-shared key) and negotiate
  which cipher, integrity algorithm, and key-exchange method to use, then derive the actual keys IPsec
  uses.

So "IKEv2/IPsec" describes one working tunnel made of two halves: IKEv2 sets it up, IPsec (via ESP)
carries the encrypted traffic once it's up. Everything below traces both halves end to end, from the
`swanctl.conf` directives that configure them, through the kernel tables that enforce them, to the
actual bytes on the wire.

## Topology

Two LANs that cannot otherwise reach each other, joined only by the tunnel across a simulated
public network (`198.51.100.0/24`, reserved for documentation by RFC 5737 so it cannot collide with
anything real):

```mermaid
flowchart LR
    onPremisesServer["<b>on-premises-server</b><br/>10.10.0.100<br/><br/><i>entrypoint in compose:</i><br/>ip route add 10.20.0.0/24<br/>via 10.10.0.20"]
    onPremisesGateway["<b>on-premises-gateway</b><br/>10.10.0.20 / 198.51.100.20<br/><br/><i>on-premises-gateway/swanctl.conf</i><br/>start_action = start<br/>(dials out)"]
    cloudGateway["<b>cloud-gateway</b><br/>198.51.100.10 / 10.20.0.10<br/><br/><i>cloud-gateway/swanctl.conf</i><br/>start_action = trap<br/>(waits, then responds)"]
    cloudServer["<b>cloud-server</b><br/>10.20.0.100<br/><br/><i>entrypoint in compose:</i><br/>ip route add 10.10.0.0/24<br/>via 10.20.0.10"]

    onPremisesServer --- onPremisesGateway
    onPremisesGateway == "IKEv2/IPsec tunnel across internet 198.51.100.0/24<br/>UDP/4500, AES256-SHA256-MODP2048" ==> cloudGateway
    cloudGateway --- cloudServer

    subgraph onPremisesLan [on-premises-lan 10.10.0.0/24]
        onPremisesServer
    end
    subgraph cloudLan [cloud-lan 10.20.0.0/24]
        cloudServer
    end
```

## How it works

IKEv2 negotiates in two phases, and `swanctl.conf` has one block per phase:

- **IKE_SA** (`proposals`) authenticates the two gateways to each other and derives keys to protect the negotiation itself. Here that's a pre-shared key (`secrets { ike-site-to-site { ... } }`).
- **CHILD_SA** (`children { lan { esp_proposals ... } }`) is what actually encrypts application traffic. `local_ts` / `remote_ts` are the traffic selectors -- only packets between those two subnets are encrypted; everything else is unaffected. `modp2048` is Diffie-Hellman group 14; including a DH group in `esp_proposals` (not just `proposals`) is specifically what turns on perfect forward secrecy for the CHILD_SA.

Once the CHILD_SA is installed, the kernel enforces it through two tables you can inspect directly:

- `ip xfrm state` -- the Security Association Database (SAD): the actual negotiated keys, cipher,
  and SPI (Security Parameter Index) for each direction.
- `ip xfrm policy` -- the Security Policy Database (SPD): which traffic (by source/destination
  subnet) gets sent through a given SA, in the `out`, `in`, and `fwd` directions.

This split -- SPD decides *what* gets protected, SAD decides *how* -- is the core of how IPsec works
underneath any tool built on top of it.

```mermaid
sequenceDiagram
    participant L as on-premises-server<br/>10.10.0.100
    participant O as on-premises-gateway<br/>kernel (SPD/SAD)
    participant C as cloud-gateway<br/>kernel (SPD/SAD)
    participant S as cloud-server<br/>10.20.0.100

    Note over O,C: Phase 1 -- IKE_SA (proposals, PSK auth)
    O->>C: IKE_SA_INIT / IKE_AUTH<br/>aes256-sha256-modp2048, PSK

    Note over O,C: Phase 2 -- CHILD_SA "lan" (esp_proposals, traffic selectors)
    O->>C: CREATE_CHILD_SA<br/>local_ts 10.10.0.0/24, remote_ts 10.20.0.0/24
    C-->>O: CHILD_SA agreed

    Note over O: kernel installs CHILD_SA:<br/>SPD (what) + SAD (how, incl. SPI per direction)
    Note over C: kernel installs the same CHILD_SA,<br/>SPIs swapped

    L->>O: plain ICMP echo request
    Note over O: SPD match -> encrypt, stamp with outbound SPI c270d7d0
    O->>C: ESP-in-UDP/4500 (ciphertext)
    Note over C: look up SPI in SAD -> decrypt
    C->>S: plain ICMP echo request

    S->>C: plain ICMP echo reply
    Note over C: SPD match -> encrypt, stamp with outbound SPI cd248e72
    C->>O: ESP-in-UDP/4500 (ciphertext)
    Note over O: look up SPI in SAD -> decrypt
    O->>L: plain ICMP echo reply
```

- **IKE** -- Internet Key Exchange. The protocol that negotiates and authenticates -- the handshake.
- **SA** -- Security Association. An agreement between the two gateways on how to protect traffic in
  one direction: which cipher, which keys, which integrity algorithm. Comes in two kinds here:
  - **IKE_SA** -- protects the negotiation itself (phase 1).
  - **CHILD_SA** -- protects the actual application traffic (phase 2); the one here is named `lan`.
- **ESP** -- Encapsulating Security Payload. The wire format that does the actual encrypting: take a
  packet, encrypt its payload, wrap it in an ESP header/trailer.
- **SPI** -- Security Parameter Index. A number stamped in every ESP packet's header that labels
  which SA it belongs to, so the receiving kernel knows which key to decrypt it with. Each direction
  of a CHILD_SA gets its own SPI.
- **SAD** -- Security Association Database (`ip xfrm state`). *How* to protect traffic: the actual
  keys, cipher, and SPI for each installed SA.
- **SPD** -- Security Policy Database (`ip xfrm policy`). *What* to protect: which traffic, by
  source/destination subnet, gets sent through which SA.
- **DH** -- Diffie-Hellman. The key-exchange method the two gateways use to agree on a shared secret
  without ever sending it over the wire. `modp2048` is DH group 14.
- **PSK** -- Pre-Shared Key. The one shared secret both gateways already know, used here to
  authenticate the IKE_SA (see Notes below for why certificates are the more common alternative).

## Notes

- **Policy-based, not route-based.** Production IPsec deployments often bind a CHILD_SA to a virtual network interface (a route-based tunnel, e.g. Linux's XFRM interfaces) so ordinary routing decides what enters the tunnel. This setup uses classic policy-based IPsec instead, where the traffic selectors themselves decide -- partly because it is the clearer first lesson (the SPD is directly visible), and partly because route-based tunnels need kernel support (`CONFIG_XFRM_INTERFACE`) that some container runtimes do not build in.
- **Pre-shared key, not certificates.** A PSK keeps this setup to one shared secret instead of a certificate authority and per-gateway certificates. Production IKEv2 deployments more often use certificate authentication, since a PSK is a single point of compromise shared by both ends.
