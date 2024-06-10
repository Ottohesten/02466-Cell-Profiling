#!/bin/sh
#BSUB -J train
#BSUB -o train_%J.out
#BSUB -e train_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=5G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 02:00
#BSUB -u otto@skytop.dk
#BSUB -B
#BSUB -N
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
module load scipy/1.11.4-python-3.10.13

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source env/bin/activate

python train.py
