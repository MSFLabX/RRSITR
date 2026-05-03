#!/bin/bash
SEEDS=(3 13 25 42 65)

for SEED in "${SEEDS[@]}"
do
    echo "===================================================="
    echo "Running Test with SEED: $SEED"
    echo "===================================================="

    CUDA_VISIBLE_DEVICES=0 \
    torchrun --nproc_per_node 1 -m training.test \
        --retrieval-data1 '/xiaolin/xyq/MyMethod/RRSITR/precomp/nwpu/nwpu_test.csv' \
        --retrieval-frequency 1 \
        --image_dir '/xiaolin/xyq/RS/Datasets/NWPU-RESISC45'\
        --retrieval-images-dir '/xiaolin/xyq/RS/Datasets/NWPU-RESISC45'\
        --datasets-dir '/xiaolin/xyq/RS/Datasets/NWPU-RESISC45' \
        --w_simcse 1.0\
        --csv-img-key filename\
        --retrieval-csv-img-key filename\
        --alpha_1 0.9 \
        --alpha_2 0.1 \
        --batch-size 100 \
        --workers 8 \
        --image-model 'ViT-B-32' \
        --image-model-builder 'openclip' \
        --text-model 'ViT-B-32' \
        --text-model-builder 'openclip' \
        --pretrained-image-model  \
        --pretrained-text-model \
        --loss 'InfoNCE' \
        --resume "/xiaolin/xyq/MyMethod/RRSITR/training/logs/NWPU/NC00/seed${SEED}/checkpoints/best_model.pt" \
        --logs "/xiaolin/xyq/MyMethod/RRSITR/training/logs/NWPU/NC00/seed${SEED}/checkpoints" \
        --name 'eval_only_result_test'
    echo "Finished evaluation for SEED: $SEED"
done

