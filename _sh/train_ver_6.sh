# HL mode sum
python train_pseudo_label.py \
  --save exp_data \
  --model_dir exp/pseudo_train_5 \
  --entropy_lambda 0.001 \
  --alpha 0.01 \
  --num_epoch 1