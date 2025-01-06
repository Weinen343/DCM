config=config/mistral/mistral-origin-64-3.yaml

datasets="narrativeqa,qasper,multifieldqa_en,\
hotpotqa,2wikimqa,musique,\
gov_report,qmsum,multi_news,\
trec,triviaqa,samsum,\
passage_count,passage_retrieval_en,\
lcc,repobench-p"

output_dir=benchmark/result_1
mkdir ${output_dir} 

python benchmark/pred.py \
--config_path ${config} \
--output_dir_path ${output_dir} \
--datasets ${datasets} \
--verbose
python benchmark/eval.py --dir_path ${output_dir}