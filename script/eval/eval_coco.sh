cd src
torchrun --nproc_per_node 2 -m \
    --master_addr=127.0.0.3 --master_port=29533 \
    training.main \
    -- \
    --eval \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --val-data="path/to/cocoir/"  \
    --csv-img-key filepath \
    --csv-caption-key title \
    --val-data-root path/to/cocoir \
    --warmup 10000 \
    --batch-size=512 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 32 \
    --workers=16 \
    --model ViT-T-16 \
    --resume path/to/model.pt \
    --logs path/to/logs/  \
    --tag eval_coco
