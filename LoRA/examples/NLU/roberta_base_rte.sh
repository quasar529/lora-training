export output_dir="./rte"
python examples/text-classification/run_glue_rte.py \
--model_name_or_path roberta-base \
--task_name rte \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 64 \
--learning_rate 5e-4 \
--num_train_epochs 80 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--apply_lora \
--lora_r 8 \
--lora_alpha 8 \
--seed 0 \
--weight_decay 0.1
