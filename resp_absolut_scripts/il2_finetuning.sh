#!/bin/bash

#SBATCH -p gpu
#SBATCH --job-name il2_evodiff
#SBATCH --output il2_evodiff
#SBATCH -w gpu-2
#SBATCH --gres=gpu:1
#SBATCH --mem=50G


module load cuda

source /stg3/data3/Jonathan/.bashrc
source /stg3/data3/Jonathan/.bash_profile

conda activate evodiff


python fine_tune.py --config_fpath /stg3/data3/Jonathan/jonathan2/resp2_absolut/evodiff_ft/config/config640M.json \
	--out_fpath /stg3/data3/Jonathan/jonathan2/resp2_absolut/fine_tuning/il2_90/ \
	--train_fpath /stg3/data3/Jonathan/jonathan2/resp2_absolut/fine_tuning/il2_90/il2_90_genAI_train.txt \
	--valid_fpath /stg3/data3/Jonathan/jonathan2/resp2_absolut/fine_tuning/il2_90/il2_90_genAI_test.txt \
    --large_model


python fine_tune.py --config_fpath /stg3/data3/Jonathan/jonathan2/resp2_absolut/evodiff_ft/config/config640M.json \
	--out_fpath /stg3/data3/Jonathan/jonathan2/resp2_absolut/fine_tuning/il2_80/ \
	--train_fpath /stg3/data3/Jonathan/jonathan2/resp2_absolut/fine_tuning/il2_80/il2_80_genAI_train.txt \
	--valid_fpath /stg3/data3/Jonathan/jonathan2/resp2_absolut/fine_tuning/il2_80/il2_80_genAI_test.txt \
    --large_model
