#!/bin/bash
SEEDS=(3 13 25 42 65)
NCS=("00" "02" "04" "06" "08")

for NC in "${NCS[@]}"
do
    for SEED in "${SEEDS[@]}"
    do
        echo "------------------------------------------------"
        echo "Starting training with SEED: $SEED | NC: $NC"
        echo "------------------------------------------------"

        CUDA_VISIBLE_DEVICES=0 \
        torchrun --nproc_per_node 1 -m  training.main \
            --train-data "/xiaolin/xyq/MyMethod/RRSITR/precomp/rsicd/RSICD-NC-${NC}.csv" \
            --retrieval-data1 '/xiaolin/xyq/MyMethod/RRSITR/precomp/rsicd/rsicd_val.csv' \
            --retrieval-frequency 1 \
            --image_dir '/xiaolin/xyq/RS/Datasets/RSICD/RSICD_images'\
            --retrieval-images-dir '/xiaolin/xyq/RS/Datasets/RSICD/RSICD_images'\
            --datasets-dir '/xiaolin/xyq/RS/Datasets/RSICD/RSICD_images' \
            --w_simcse 1.0\
            --csv-img-key filename\
            --retrieval-csv-img-key filename\
            --gamma_1 5 \
            --gamma_2 18 \
            --lambda_1 0.8 \
            --lambda_2 0.9 \
            --sigma 0.6 \
            --alpha_1 0.9 \
            --alpha_2 0.1 \
            --epochs 50 \
            --seed $SEED \
            --save-frequency 0 \
            --batch-size 100 \
            --workers 8 \
            --lr 7e-06 \
            --warmup 200 \
            --weight_decay 0.7 \
            --max-grad-norm 50.0 \
            --image-model 'ViT-B-32'\
            --image-model-builder 'openclip' \
            --text-model 'ViT-B-32' \
            --text-model-builder 'openclip'\
            --pretrained-image-model  \
            --pretrained-text-model \
            --loss 'InfoNCE' \
            --report-to tensorboard \
            --logs "/xiaolin/xyq/MyMethod/RRSITR/training/logs/RSICD/NC${NC}"  \
            --name "seed${SEED}"

        echo "Finished training for SEED: $SEED | NC: $NC"
    done
done
