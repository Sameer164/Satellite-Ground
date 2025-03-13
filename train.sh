echo "Starting training..."
python train.py \
    --backbone res50 \
    --batch_size 32 \
    --epochs 20 \

echo "Training complete!"