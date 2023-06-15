#!/bin/sh
#SBATCH -J homps100
#SBATCH -o ./N.out
#SBATCH -D ./
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --get-user-env
#SBATCH --mail-type=end
#SBATCH --mem=10mb
#SBATCH --mail-user=benjamin.sappler@tum.de
#SBATCH --export=NONE
#SBATCH --time=12:00:00
module load slurm_setup
module load python
source ../../../HOPS_env/bin/activate
python script.py
source ../../../HOPS_env/bin/deactivate