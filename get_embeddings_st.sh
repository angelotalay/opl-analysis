#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --account=talayag-agt-computations
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=36
#SBATCH --time=12:0:0
#SBATCH --mail-type=ALL

set -e

module purge; module load bluebear
module load bear-apps/2022b
module load Python/3.10.8-GCCcore-12.2.
module load PyTorch/2.1.2-foss-2022b-CUDA-12.0.0

export VENV_PATH="/rds/homes/t/talayag/talayag-agt-computations/opl_analysis/opl-analysis-venv-${BB_CPU}"
SCRIPT_PATH="/rds/homes/t/talayag/talayag-agt-computations/opl_analysis/analysis-files"

# Activate the virtual environment
source ${VENV_PATH}/bin/activate


# Execute your Python script with the script path
python ${SCRIPT_PATH}/calculate_embeddings_st.py

