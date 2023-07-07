# Put your WANDB API key here to enable logging to wandb.
export WANDB_API_KEY='my wandb key'

python3 -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,-1,1' \
    --total_steps=10000 \
    --tokenizer_path='tokenizer/32k' \
    --data_path='gcs://donggyukimc/pretrain_tokens/0' \
    --output_dir='output' \
    --dtype='fp16' \
    --batch_size 2 \
    --seq_length 4096 \
    --save_model_freq=1000 \
    --save_milestone_freq=1000 \
    --load_llama_config='tiny' \
    --optimizer.type='adamw' \
    --optimizer.accumulate_gradient_steps=1 \
    --optimizer.adamw_optimizer.weight_decay=0.01 \
    --optimizer.adamw_optimizer.lr=3e-4 \
    --optimizer.adamw_optimizer.end_lr=3e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=2000 \
    --optimizer.adamw_optimizer.lr_decay_steps=250000 \
    --optimizer.adamw_optimizer.multiply_by_parameter_scale=True \
    --optimizer.adamw_optimizer.bf16_momentum=True \
    --logger.online=True \
    --logger.project="test" \
    --logger.output_dir="output"