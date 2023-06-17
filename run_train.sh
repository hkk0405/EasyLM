python3 -m EasyLM.models.llama.llama_train \
    --tokenizer_path='EleutherAI/polyglot-ko-1.3b' \
    --data_path='output/features' \
    --output_dir='output' \
    --dtype='fp16' \
    --save_model_freq=1000 \
    --save_milestone_freq=1000 \
    --load_llama_config='tiny'