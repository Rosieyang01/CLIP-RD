cd src
torchrun --nproc_per_node 1 -m \
    --master_addr=127.0.0.3 --master_port=29560 \
    training.main \
    -- \
    --eval \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --imagenet-val=path/to/imageNet_val/ \
    --imagenet-v2=path/to/imagenetv2-matched-frequency-format-val_copy/ \
    --imagenet-r=path/to/imagenet-r/ \
    --imagenet-sketch=path/to/sketch/ \
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
