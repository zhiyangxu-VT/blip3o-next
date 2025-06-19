#!/bin/bash

python -m unifork.eval.model_vqa_loader \
    --model-path your/model/path \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/MME/answers/unifork.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment unifork

cd eval_tool

python calculation.py --results_dir answers/unifork
