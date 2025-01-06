config=config/llama/llama-origin-64-3.yaml
#config/mistral/mistral-origin-64-1.yaml
#HuggingFaceTB/cosmopedia-20k
#HuggingFaceTB/cosmopedia_6M
#jayasuryajsk/Long_text
#THUDM/LongBench
python pretrain.py \
--datasets THUDM/LongWriter-6k \
--learning_rate 1e-5 \
--data_type text \
--num_train_epochs 1 \
--freeze_llm True \
--config_path $config 




