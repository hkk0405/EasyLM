# Put your WANDB API key here to enable logging to wandb.
export WANDB_API_KEY='my wandb key'

# TPU specific flags to improve training throughput
export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --jax_enable_async_collective_offload=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'

python3 -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,-1,1' \
    --total_steps=10000000 \
    --tokenizer_path='tokenizer/32k' \
    --data_path='gcs://donggyukimc/pretrain_tokens/0' \
    --output_dir='gs://donggyukimc/checkpoint' \
    --dtype='fp16' \
    --batch_size 64 \
    --seq_length 4096 \
    --save_model_freq=100000 \
    --save_milestone_freq=100000 \
    --load_llama_config='3b' \
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