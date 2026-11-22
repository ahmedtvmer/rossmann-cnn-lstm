#!/bin/sh
# Reproduces every result. Each training run takes about 1 minute on CPU.
set -e

# 1) Rossmann: the paper's models, the baseline and the three bonus candidates, 3 seeds each
for seed in 42 43 44; do
  for model in s_cnn_lstm p_cnn_lstm lstm; do
    python train.py --model $model --seed $seed
  done
  python train.py --model p_cnn_lstm --seed $seed --next-day-features   # candidate (a)
  python train.py --model p_cnn_lstm --seed $seed --lags 14             # candidate (b)
  python train.py --model p_cnn_lstm --seed $seed --log-sales           # candidate (c)
done
python evaluate.py --split val
# -> (a) wins on validation. Move the rejected candidates aside, then score the test set once:
mkdir -p results/rossmann/rejected_candidates
mv results/rossmann/models/p_cnn_lstm_lags14_* results/rossmann/models/p_cnn_lstm_log_* \
   results/rossmann/rejected_candidates/
python evaluate.py --split test

# 2) The paper's dataset: same models plus the chosen change, to compare with the paper itself
for seed in 42 43 44; do
  for model in s_cnn_lstm p_cnn_lstm lstm; do
    python train.py --model $model --seed $seed --dataset paper
  done
  python train.py --model p_cnn_lstm --seed $seed --dataset paper --next-day-features
done
python evaluate.py --split test --dataset paper
