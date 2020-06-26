#~/bin/bash


python inference.py \
    --dataDir ./data \
    --cls  mujoco_data \
    --model ./log/overfit1/best_net_198.pth \
    --outf ./log/overfit1_recon \
    --type test \
    --num_points 2500 \
    --chamLoss_wt \
