echo "Starting training..."
python train.py \
    --batch_size 128 \
    --epochs 100 \
    --num_workers 4 \
    --pin_memory \
    --use_contrastive

echo "Training complete!"