"""Build test sequences with the exact harness recipe (test events: every CDM has ttca >= 2 d). No caching into data/."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import numpy as np, pandas as pd, harness as H
te = H.clean(pd.read_csv(H.RAW + "test_data.csv"))
inp = te[te.time_to_tca >= 2]
eids = np.sort(inp.event_id.unique())
X, M = H.build_sequences(inp, eids)
np.savez_compressed(os.path.join(HERE, "test_sequences.npz"), X=X, mask=M, lengths=M.sum(1),
                    event_key=np.array([f"test:{e}" for e in eids]), feature_names=np.array(H.SEQ_COLS))
print("test sequences", X.shape, "events", len(eids), "all CDMs >= 2 d:", bool((te.time_to_tca >= 2).all()))
