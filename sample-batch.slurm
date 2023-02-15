## Rename to batch.slurm
## To run in terminal navigate to this directory and run command 'sbatch batch.slurm'
## Insert account name, partition name, and email to use. Also update PATH to 
## be your locally installed Anaconda path, and activate your correct environment.

#!/bin/bash
#SBATCH --account=INSERT_ACCOUNT
#SBATCH --partition=INSERT_PARTITION
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time 14-00:00:00
#SBATCH --mem=32gb
#SBATCH --mail-user=INSERT_EMAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=./slurm/output_%j.out
##SBATCH --error=./error_%j.err

## CHECK FILE NAME!
export PATH="INSERT_ANACONDA_PATH"
conda activate INSERT_YOUR_ENVIRONMENT
python ToxiCR.py --algo BERT --embed bert --keyword --mode pretrain