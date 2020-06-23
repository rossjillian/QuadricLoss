#~/bin/bash


python inference.py \
    --dataDir ./data \
    --cls  mujoco_data \
    --model ./out/best_net_140.pth \
    --outf ./out \
    --type test \
    --num_points 2500 \
    --chamLoss_wt \
