#!/bin/bash

python -m unifork.eval.model_vqa_loader \
    --model-path your/model/path \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/unifork.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python unifork/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/unifork.jsonl
