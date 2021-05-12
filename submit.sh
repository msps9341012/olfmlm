#!/bin/bash
#

#SBATCH --output=/iesl/canvas/rueiyaosun/logs/res_%j.txt  # output file
#SBATCH -e /iesl/canvas/rueiyaosun/logs/res_%j.err # File to which STDERR will be written
#SBATCH --partition=gpu # Partition to submit to 
#SBATCH --gres=gpu:1
#SBATCH --time=07-00:00      # Runtime in D-HH:MM
#SBATCH --mem=20G    # Memory in MB per cpu allocated
#SBATCH --exclude='gpu-0-0, gpu-0-1, gpu-0-2'

cd /iesl/canvas/rueiyaosun
python -m olfmlm.evaluate.main --exp_name mf+mlm --overrides "run_name = "${1}_1", lr=5e-5"

python -m olfmlm.evaluate.main --exp_name mf+mlm --overrides "run_name = "${1}_2", lr=3e-5"

python -m olfmlm.evaluate.main --exp_name mf+mlm --overrides "run_name = "${1}_3", lr=2e-5"

exit


