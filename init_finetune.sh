#!/bin/bash

#SBATCH -p gpu
#SBATCH --job-name l1evodiff
#SBATCH --output l1evodiff
#SBATCH -w gpu-3
#SBATCH --gres=gpu:1
#SBATCH --mem=50G

module load cuda

source /stg3/data3/Jonathan/.bashrc
source /stg3/data3/Jonathan/.bash_profile

conda activate evodiff


python small_model_train.py --config_fpath /stg3/data3/Jonathan/jonathan2/pdl1_evodiff_tuning/evodiff/config/config640M.json \
	--out_fpath /stg3/data3/Jonathan/jonathan2/pdl1_evodiff_tuning/evodiff_results/ \
	--train_fpath /stg3/data3/Jonathan/jonathan2/pdl1_evodiff_tuning/train_pdl1.txt \
	--valid_fpath /stg3/data3/Jonathan/jonathan2/pdl1_evodiff_tuning/test_pdl1.txt \
        --checkpoint_freq 60 \
	--large_model
