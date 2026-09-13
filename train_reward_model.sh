#!/bin/bash

set -e

uv add hf-transfer

dataset_name="Anthropic/hh-rlhf"
model_name="HuggingFaceTB/SmolLM-135M"
dataset_dir="dataset/hh-rlhf"
model_dir="model/SmolLM-135M"

if [ ! -d $dataset_dir ];
then
	echo "Downloading Dataset"
	uvx hf download $dataset_name --local-dir $dataset_dir --repo-type dataset &
fi

if [ ! -d $model_dir ];
then
	echo "Downloading Model"
	uvx hf download  $model_name --local-dir $model_dir &
fi

wait


uv run reward_model/main.py $model_dir 'model/reward_checkpoint/' $dataset_dir reward_model/template.jinja
