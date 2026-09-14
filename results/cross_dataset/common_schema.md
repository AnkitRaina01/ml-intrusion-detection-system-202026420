# Common flow schema — CIC-IDS2017 ↔ UNSW-NB15 alignment

> This is the **documented transformation** required by the Phase-4 brief: the two datasets are NOT forced into an identical feature space silently. NSL-KDD is excluded from this alignment because it has no comparable bidirectional-flow timing features.

Both datasets describe bidirectional network flows, but they are built by different tools (CIC-IDS2017: CICFlowMeter; UNSW-NB15: Argus + Bro/Zeek) with different units and slightly different definitions. The 12 features below are the subset that can be constructed the same way on both sides. Derived rates are **recomputed from the primitives** on both datasets so the definition is identical; CIC-IDS2017 `Flow Duration` (microseconds) is divided by 1e6 to match UNSW `dur` (seconds). Non-finite results (division by zero duration) are set to 0.

| common feature | from CIC-IDS2017 | from UNSW-NB15 |
|---|---|---|
| `dur_s` | Flow Duration / 1e6  (µs -> s) | dur  (already seconds) |
| `fwd_pkts` | Total Fwd Packets | spkts |
| `bwd_pkts` | Total Backward Packets | dpkts |
| `fwd_bytes` | Total Length of Fwd Packets | sbytes |
| `bwd_bytes` | Total Length of Bwd Packets | dbytes |
| `fwd_pkt_len_mean` | Fwd Packet Length Mean | smean |
| `bwd_pkt_len_mean` | Bwd Packet Length Mean | dmean |
| `flow_bytes_per_s` | (fwd_bytes+bwd_bytes)/dur_s  [recomputed] | (sbytes+dbytes)/dur  [recomputed] |
| `flow_pkts_per_s` | (fwd_pkts+bwd_pkts)/dur_s  [recomputed] | (spkts+dpkts)/dur  [recomputed] |
| `fwd_pkts_per_s` | fwd_pkts/dur_s  [recomputed] | spkts/dur  [recomputed] |
| `bwd_pkts_per_s` | bwd_pkts/dur_s  [recomputed] | dpkts/dur  [recomputed] |
| `down_up_ratio` | bwd_pkts/max(fwd_pkts,1)  [recomputed] | dpkts/max(spkts,1)  [recomputed] |

## Why this is lossy (limitations)

- **Different exporters.** CICFlowMeter and Argus segment flows, count packets, and handle timeouts differently, so the *same* named quantity is not measured identically.
- **`fwd_pkt_len_mean` / `bwd_pkt_len_mean`** are taken dataset-native (CIC `Fwd/Bwd Packet Length Mean` vs UNSW `smean`/`dmean`); these are close in intent but not guaranteed identical in definition.
- **No protocol / service / TCP-flag features** are shared (UNSW `proto` has 130+ values; CIC encodes flags as counts). The common schema is purely volume/rate/duration.
- **Label semantics differ**: CIC-IDS2017 attack mix (DoS/DDoS/PortScan/brute-force/web/bot/infiltration) is not the UNSW-NB15 mix (Exploits/Fuzzers/Generic/Recon/…). Only the binary benign-vs-attack target is aligned.

Common-schema sample sizes actually used: CIC-IDS2017 train 102,571 / test 30,000; UNSW-NB15 train 58,924 / test 82,332.
