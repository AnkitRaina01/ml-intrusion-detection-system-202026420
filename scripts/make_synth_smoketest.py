"""
Standalone synthetic-data generator for Phase-1 smoke testing.
Replicates a CICIDS-like schema with a configurable row count so the inherited
SHAP / real-time module (which assumes a large dataset) can be exercised
end-to-end.

This produces RANDOM NOISE, not real network traffic. It is ONLY a plumbing
smoke test. No metric produced from it is a real experimental result.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/raw/synthetic_smoketest.csv")
np.random.seed(42)

data = {
    'Flow_Duration': np.random.randint(0, 1000000, n_samples),
    'Total_Fwd_Packets': np.random.randint(1, 100, n_samples),
    'Total_Backward_Packets': np.random.randint(1, 100, n_samples),
    'Total_Length_of_Fwd_Packets': np.random.randint(0, 50000, n_samples),
    'Total_Length_of_Bwd_Packets': np.random.randint(0, 50000, n_samples),
    'Flow_Bytes/s': np.random.uniform(0, 100000, n_samples),
    'Flow_Packets/s': np.random.uniform(0, 1000, n_samples),
    'Flow_IAT_Mean': np.random.uniform(0, 10000, n_samples),
    'Flow_IAT_Std': np.random.uniform(0, 5000, n_samples),
    'Fwd_IAT_Mean': np.random.uniform(0, 10000, n_samples),
    'Bwd_IAT_Mean': np.random.uniform(0, 10000, n_samples),
    'Fwd_PSH_Flags': np.random.randint(0, 5, n_samples),
    'Bwd_PSH_Flags': np.random.randint(0, 5, n_samples),
    'Fwd_Header_Length': np.random.randint(20, 60, n_samples),
    'Bwd_Header_Length': np.random.randint(20, 60, n_samples),
    'Min_Packet_Length': np.random.randint(0, 100, n_samples),
    'Max_Packet_Length': np.random.randint(100, 1500, n_samples),
    'Packet_Length_Mean': np.random.uniform(100, 800, n_samples),
    'Packet_Length_Std': np.random.uniform(0, 500, n_samples),
    'FIN_Flag_Count': np.random.randint(0, 3, n_samples),
    'SYN_Flag_Count': np.random.randint(0, 3, n_samples),
    'RST_Flag_Count': np.random.randint(0, 3, n_samples),
    'ACK_Flag_Count': np.random.randint(0, 50, n_samples),
    'URG_Flag_Count': np.random.randint(0, 2, n_samples),
    'Average_Packet_Size': np.random.uniform(100, 800, n_samples),
    'Init_Win_bytes_forward': np.random.randint(0, 65535, n_samples),
    'Init_Win_bytes_backward': np.random.randint(0, 65535, n_samples),
    'Active_Mean': np.random.uniform(0, 10000, n_samples),
    'Active_Std': np.random.uniform(0, 5000, n_samples),
    'Idle_Mean': np.random.uniform(0, 10000, n_samples),
    'Idle_Std': np.random.uniform(0, 5000, n_samples),
}
labels = np.random.choice(['BENIGN', 'DDoS', 'PortScan', 'Bot'],
                          n_samples, p=[0.80, 0.10, 0.05, 0.05])
data['Label'] = labels
df = pd.DataFrame(data)

# inject 5% missing + 2% duplicates like the repo's own test harness
mask = np.random.random(df.shape) < 0.05
df = df.mask(mask)
n_dup = int(n_samples * 0.02)
dup_idx = np.random.choice(df.index, n_dup, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"wrote {out}  rows={len(df):,}  cols={len(df.columns)}  "
      f"missing={int(df.isnull().sum().sum()):,}  dup={int(df.duplicated().sum()):,}")
print(df['Label'].value_counts().to_dict())
