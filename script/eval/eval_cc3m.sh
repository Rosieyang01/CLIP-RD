cd src
torchrun --nproc_per_node 1 -m \
    --master_addr=127.0.0.3 --master_port=29533 \
    training.main \
    -- \
    --eval \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="path/to/cc3m_train.csv,path/to/cc12m.csv"  \
    --val-data="path/to/cc3m/val"  \
    --data-root path/to/cc3m/images/,path/to/cc12m/images/ \
    --val-data-root path/to/cc3m_val \
    --csv-img-key filepath \
    --csv-caption-key title \
    --warmup 10000 \
    --batch-size=1024 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 32 \
    --workers=16 \
    --model ViT-T-16 \
    --resume path/to/model.pt \
    --logs path/to/logs/  \
    --tag eval_cc3m_val
