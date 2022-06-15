#!/bin/bash
python3 generate.py	/home/gb/dlc/python/DLMulMix-main/data-bin/test2016 \
				--path /home/gb/dlc/python/DLMulMix-main/results/en-de/mmtimg/checkpoint83.pt \
				--source-lang en --target-lang de \
				--beam 5 \
				--num-workers 12 \
				--batch-size 128 \
				--results-path results/pre_mixup/mmtimg/results2016 \
				--remove-bpe \

