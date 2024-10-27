# HL mode sum
python train_midterm_checkpoint.py \
  --save exp_data \
  --model_dir exp/pseudo_train_8 \
  --entropy_lambda 0.001 \
  --rotate_lambda 0.001 \
  --alpha 0.01 \
  --num_epoch 1 \
  --reconstruction_lambda 0 \
