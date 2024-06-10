#!/bin/sh
#BSUB -J torch_gpu
#BSUB -o torch_gpu_%J.out
#BSUB -e torch_gpu_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=5G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 5
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
module load scipy/1.11.4-python-3.10.13

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source env/bin/activate

python matmul_torch.py
