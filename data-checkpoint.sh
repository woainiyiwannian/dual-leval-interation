#!/bin/bash


python3 /home/gb/dlc/python/DLMulMix-mainyuanban/scripts/average_checkpoints.py \
			--inputs /home/gb/dlc/python/DLMulMix-mainyuanban/results/premix10/mmtimg10 \
			--num-epoch-checkpoints 18 \
			--output /home/gb/dlc/python/DLMulMix-mainyuanban/results/premix10/mmtimg10/model.pt \
