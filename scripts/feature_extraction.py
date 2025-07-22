
import pyshark
import pandas as pd
import numpy as np
from collections import defaultdict

def extract_features_from_pcap(pcap_file, output_csv):
    print(f"🔍 Reading from PCAP: {pcap_file}")
    cap = pyshark.FileCapture(pcap_file, use_json=True, include_raw=True)

    flows = defaultdict(list)

    for pkt in cap:
        print("📦 Packet:", pkt)  # DEBUG LINE
        try:
            if 'IP' in pkt:
                src_ip = pkt.ip.src
                dst_ip = pkt.ip.dst

                # Detect protocol safely
                if hasattr(pkt, 'transport_layer') and pkt.transport_layer:
                    proto = pkt.transport_layer
                elif hasattr(pkt, 'tcp'):
                    proto = 'TCP'
                elif hasattr(pkt, 'udp'):
                    proto = 'UDP'
                else:
                    proto = 'UNKNOWN'

                # Packet length
                try:
                    length = int(pkt.length)
                except AttributeError:
                    length = len(pkt.get_raw_packet())

                timestamp = float(pkt.sniff_timestamp)

                flow_key = (src_ip, dst_ip, proto)
                flows[flow_key].append((timestamp, length))
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    cap.close()

    if not flows:
        print("⚠️ No flows found in PCAP.")
        return

    rows = []
    for flow_key, packets in flows.items():
        if len(packets) < 2:
            continue

        timestamps, lengths = zip(*packets)
        iats = np.diff(timestamps)

        features = [
            len(lengths),                    # total_packets
            sum(lengths),                    # total_bytes
            min(lengths),                    # min_pkt_len
            max(lengths),                    # max_pkt_len
            np.mean(lengths),                # mean_pkt_len
            np.std(lengths),                 # std_pkt_len
            timestamps[-1] - timestamps[0],  # flow_duration
            min(iats),
            max(iats),
            np.mean(iats),
            np.std(iats),
        ]

        padded_features = features + [0] * (84 - len(features))
        rows.append(padded_features)

    if not rows:
        print("⚠️ No valid flows with ≥2 packets found.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, header=False)
    print(f"✅ Extracted {len(rows)} flows → {output_csv}")

# Run
if __name__ == "__main__":
    extract_features_from_pcap(
        "/home/xold/ids-project/data/capture.pcap",
        "/home/xold/ids-project/data/features.csv"
    )
