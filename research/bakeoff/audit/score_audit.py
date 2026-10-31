"""AUDIT: run the unmodified score.py main() over audit/oof (H.OOF redirected). Writes audit/leaderboard_audit.{md,json}."""
import sys, os
BK = "/root/orbital-dodge/research/bakeoff"; sys.path.insert(0, BK)
import harness as H
H.OOF = os.path.join(BK, "audit", "oof")
import score
sys.argv = ["score.py", "--splits", sys.argv[1] if len(sys.argv) > 1 else "mission", "--out", os.path.join(BK, "audit", "leaderboard_audit.md")]
score.main()
