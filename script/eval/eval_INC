cd src

torchrun --nproc_per_node 1 -m \
    --master_addr=127.0.0.3 --master_port=29560 \
    training.main -- \
    --eval \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --imagenet-c1=/path/to/imagenet-c/severity_1 \
    --imagenet-c2=/path/to/imagenet-c/severity_2 \
    --imagenet-c3=/path/to/imagenet-c/severity_3 \
    --imagenet-c4=/path/to/imagenet-c/severity_4 \
    --imagenet-c5=/path/to/imagenet-c/severity_5 \
    --warmup 10000 \
    --batch-size=1024 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 32 \
    --workers=16 \
    --model model(e.g. ViT-B-16) \
    --resume path/to/model.pt \
    --logs path/to/logs/  \
    --tag eval_IN_c

#For ImageNet-C evaluation, organize the dataset into five separate folders corresponding to severity levels 1–5. Then, provide the path to each folder using --imagenet-c1 through --imagenet-c5.
