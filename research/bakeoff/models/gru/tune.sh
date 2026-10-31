#!/bin/bash
# inner-split-only config comparison (selection criterion fixed beforehand: mean inner-val class-weighted CE, by-mission split)
cd /root/orbital-dodge/research/bakeoff
declare -A C
C[g64_lr2e3]='{"arch":"gru","hidden":64,"lr":2e-3}'
C[g64_lr5e4]='{"arch":"gru","hidden":64,"lr":5e-4}'
C[g32_lr5e4]='{"arch":"gru","hidden":32,"lr":5e-4}'
C[g64x2_lr5e4]='{"arch":"gru","hidden":64,"layers":2,"lr":5e-4}'
C[g64_lr5e4_wd1e1]='{"arch":"gru","hidden":64,"lr":5e-4,"weight_decay":0.1,"dropout":0.3}'
C[tf64_lr5e4]='{"arch":"tf","hidden":64,"layers":2,"lr":5e-4}'
i=0
for tag in "${!C[@]}"; do
  for split in mission event; do
    gpu=$([ $((i % 2)) -eq 0 ] && echo 1 || echo 3)
    GRU_CFG="${C[$tag]}" CUDA_VISIBLE_DEVICES=$gpu nohup .venv/bin/python models/gru/gru.py tune $split $tag > logs/gru_tune_${tag}_${split}.log 2>&1 &
    i=$((i+1))
  done
done
wait
echo ALLDONE
