#!/bin/bash

#SBATCH -p gpu
#SBATCH --job-name neura_evodiff
#SBATCH --output neura_evodiff
#SBATCH -w gpu-2
#SBATCH --gres=gpu:1
#SBATCH --mem=50G

module load cuda

source /stg3/data3/Jonathan/.bashrc
source /stg3/data3/Jonathan/.bash_profile

conda activate evodiff


python fine_tune.py --config_fpath /stg3/data3/Jonathan/jonathan2/resp2_absolut/evodiff_ft/config/config640M.json \
	--out_fpath /stg3/data3/Jonathan/jonathan2/resp2_absolut/fine_tuning/neura_80/ \
	--train_fpath /stg3/data3/Jonathan/jonathan2/resp2_absolut/fine_tuning/neura_80/neuraminidase_80_genAI_train.txt \
	--valid_fpath /stg3/data3/Jonathan/jonathan2/resp2_absolut/fine_tuning/neura_80/neuraminidase_80_genAI_test.txt \
    --large_model

python fine_tune.py --config_fpath /stg3/data3/Jonathan/jonathan2/resp2_absolut/evodiff_ft/config/config640M.json \
	--out_fpath /stg3/data3/Jonathan/jonathan2/resp2_absolut/fine_tuning/neura_90/ \
	--train_fpath /stg3/data3/Jonathan/jonathan2/resp2_absolut/fine_tuning/neura_90/neuraminidase_90_genAI_train.txt \
	--valid_fpath /stg3/data3/Jonathan/jonathan2/resp2_absolut/fine_tuning/neura_90/neuraminidase_90_genAI_test.txt \
    --large_model
