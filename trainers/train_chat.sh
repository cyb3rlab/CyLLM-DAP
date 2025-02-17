lr=1e-4
lora_rank=32
lora_alpha=64
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

torchrun --nnodes 1 --nproc_per_node 1 run_clm_peft_chat.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --dataset_name khangmacon/cyberQA \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --do_eval \
    --validation_split_percentage 0.1 \
    --evaluation_strategy "epoch" \
    --num_train_epochs 10 \
    --logging_dir "./logs" \
    --logging_strategy "steps" \
    --report_to "wandb" \
    --logging_steps 10 \
    --preprocessing_num_workers 4 \
    --attn_implementation "flash_attention_2" \
    --bf16 \
    --tf32 True \
    --block_size 1024 \
    --save_strategy "epoch" \
    --output_dir "./chat_llama3_base" \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing \
    --overwrite_output_dir \
    --logging_first_step True \
    --gradient_checkpointing \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout}