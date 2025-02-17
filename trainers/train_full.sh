lr=3e-5
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05

torchrun --nnodes 1 --nproc_per_node 1 run_clm_peft.py \
    --model_name_or_path mistralai/Mistral-7B-v0.3 \
    --push_to_hub \
    --hub_id "khangmacon/llmtrain_mistral" \
    --dataset_name "khangmacon/myllm" \
    --is_proccessed False \
    --save_to_disk \
    --use_cache_data \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --do_eval \
    --validation_split_percentage 0.0001 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --num_train_epochs 1 \
    --logging_dir "./logs" \
    --logging_strategy "steps" \
    --report_to "wandb" \
    --logging_steps 10 \
    --preprocessing_num_workers 8 \
    --attn_implementation "flash_attention_2" \
    --bf16 \
    --tf32 True \
    --block_size 1024 \
    --save_total_limit 2 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --output_dir "./cyllama3" \
    --base_data_dir "./train_data" \
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