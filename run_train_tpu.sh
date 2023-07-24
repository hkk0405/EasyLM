# Put your WANDB API key here to enable logging to wandb.
export WANDB_API_KEY='2abafd8692ce7bd26923f6d2a27b52b17a2bb7ef'

# TPU specific flags to improve training throughput
#export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --jax_enable_async_collective_offload=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'

python3 -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,-1,1' \
    --total_steps=10000000 \
    --tokenizer_path='tokenizer/32k' \
    --data_path='data' \
    --output_dir='gs://hcshiin/alibi_test' \
    --dtype='bf16' \
    --preprocessing_num_workers 4 \
    --batch_size 16 \
    --seq_length 4096 \
    --log_freq 100 \
    --save_model_freq=100000 \
    --save_milestone_freq=100000 \
    --load_llama_config='1b' \
    --optimizer.type='adamw' \
    --optimizer.accumulate_gradient_steps=4 \
    --optimizer.adamw_optimizer.weight_decay=0.1 \
    --optimizer.adamw_optimizer.lr=3e-4 \
    --optimizer.adamw_optimizer.end_lr=3e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=10000 \
    --optimizer.adamw_optimizer.lr_decay_steps=250000 \
    --checkpointer.save_optimizer_state=True \
    --logger.online=True \
    --logger.project="test" \
    --logger.output_dir="output" \
    --bucket_project_name sackoh