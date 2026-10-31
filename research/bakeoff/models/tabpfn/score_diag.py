"""Score DIAGNOSTIC TabPFN variants (not bake-off entrants) with the unmodified pre-registered scorer.
Points score.py at models/tabpfn/diag/ (persistence OOF symlinked in) so nothing extra lands in oof/."""
import os, sys, glob
BK = "/root/orbital-dodge/research/bakeoff"; sys.path.insert(0, BK)
import score, harness as H
D = os.path.join(BK, "models/tabpfn/diag")
for s in ("event", "mission"):
    src, dst = os.path.join(H.OOF, f"persistence_{s}.parquet"), os.path.join(D, f"persistence_{s}.parquet")
    if not os.path.exists(dst): os.symlink(src, dst)
splits = sorted({os.path.basename(f).rsplit("_", 1)[1][:-8] for f in glob.glob(os.path.join(D, "tabpfn_*_*.parquet"))})
H.OOF = D
sys.argv = ["score.py", "--splits", ",".join(splits), "--out", os.path.join(D, "leaderboard_diag.md")]
score.main()
