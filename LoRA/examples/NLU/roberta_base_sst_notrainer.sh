export TASK_NAME=sst2

python examples/text-classification/run_glue_no_trainer.py \
  --model_name_or_path textattack/roberta-base-SST-2 \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 128 \
  --learning_rate 2e-5 \
  --num_train_epochs 30 \
  --output_dir /tmp/$TASK_NAME/ \
  --do_eval \
  --lora_r 16