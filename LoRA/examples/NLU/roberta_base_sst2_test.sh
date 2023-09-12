python examples/text-classification/run_glue.py \
--model_name_or_path textattack/roberta-base-SST-2 \
--lora_path /home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/lora/2023-09-11_16:37:03lorackpt.bin \
--task_name sst2 \
--do_eval \
--output_dir ./output \
--apply_lora \
--lora_r 128 \
--lora_alpha 128
#--lora_path /home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/r64/model/checkpoint-8959/pytorch_model.bin \
#/home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/r32/model/checkpoint-30566/pytorch_model.bin
#--lora_path /home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/rank32/model/checkpoint-14229/pytorch_model.bin \
#/home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/r4/model/checkpoint-15283/pytorch_model.bin
#/home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/r16/model/checkpoint-9486/pytorch_model.bin
#/home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/r64r64r64/model/checkpoint-13175/pytorch_model.bin
# --lora_path /home/lab/bumjun/lora-training/LoRA/examples/NLU/sst2/r16/model/checkpoint-9486/pytorch_model.bin \
