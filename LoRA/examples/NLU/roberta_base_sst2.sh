# export num_gpus=2
# export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
# export PYTHONHASHSEED=0
export output_dir="./sst2/r16r128_no_merged"
python examples/text-classification/run_glue.py \
--model_name_or_path textattack/roberta-base-SST-2 \
--lora_path /home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/lora/2023-09-11_16:37:03lorackpt.bin \
--task_name sst2 \
--do_eval \
--do_train \
--max_seq_length 128 \
--per_device_train_batch_size 64 \
--learning_rate 9e-5 \
--num_train_epochs 30 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--seed 0 \
--apply_lora \
--lora_r 128 \
--lora_alpha 128 \
--weight_decay 0.1
#--lora_path /home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/r32/model/checkpoint-30566/pytorch_model.bin \
#--lora_path /home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/r64/model/checkpoint-8959/pytorch_model.bin \
#/home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/r16/model/checkpoint-9486/pytorch_model.bin
#--lora_path /home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/rank32/model/checkpoint-14229/pytorch_model.bin \
#/home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/r16/model/checkpoint-9486/pytorch_model.bin
#--lora_path /home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/r16/model/checkpoint-9486/pytorch_model.bin \