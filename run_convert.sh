python3 -m EasyLM.models.llama.convert_easylm_to_hf \
    --tokenizer_path='output/tokenizer' \
    --load_checkpoint='params::output/streaming_params' \
    --output_dir='output/converted' \
    --model_size='tiny'