#!/bin/bash

#SBATCH -c8 --mem=16gb
#SBATCH --gres gpu:1
#SBATCH -p cs
#SBATCH --nodelist=tamar


conda activate leaf
python -c "import sys; print(sys.executable)"

jupyter notebook --no-browser --port=8888 --ip=0.0.0.0