#!/bin/bash
#SBATCH --nodes=2               # number of nodes
#SBATCH --partition=primary     # GPU partition
#SBATCH --ntasks-per-node=2     # number of tasks per node
#SBATCH --cpus-per-task=8       # number of cpus requested per task
#SBATCH --threads-per-core=1    # number of threads per core
#SBATCH --gres=gpu:2            # number of gpus per node
#SBATCH --time=12:00:00         # format: HH:MM:SS
##SBATCH --nodelist=dgx06,dgx08
##SBATCH --mem 0
#SBATCH --job-name=ddpslurm
#SBATCH -o workspace/%x.output%j
#SBATCH -e workspace/%x.error%j
##SBATCH --gres-flags=enforce-binding
#SBATCH --gpu-bind=none
#SBATCH --overcommit
#SBATCH --parsable

#--nodes=2  --ntasks-per-node=4 --cpus-per-task=8 --threads-per-core=1 --gres=gpu:4 
#--nodes=2  --ntasks-per-node=8 --cpus-per-task=8 --threads-per-core=1 --gres=gpu:8 -missing gpus from node on timeline view
#--nodes=8  --ntasks-per-node=1 --cpus-per-task=8 --threads-per-core=1 --gres=gpu:1 -works
#--nodes=4  --ntasks-per-node=2 --cpus-per-task=8 --threads-per-core=1 --gres=gpu:2 -works
#--nodes=2  --ntasks-per-node=2 --cpus-per-task=8 --threads-per-core=1 --gres=gpu:2 -works
# To run : sbatch workspace/source_code/slurm/fsdp_multinode.slurm

echo "NODELIST="${SLURM_NODELIST}
echo "WORLD_SIZE="${SLURM_NPROCS}

# QdstrmImporter -i multinode_nvtx.qdstrm 
# load conda env

#module load conda/2023
#conda init
#source activate env_mnode

module load nsight-systems/2024.5.1

export TMPDIR=$PWD/workspace/reports/tmp
echo $TMPDIR
echo $PWD

# get master node IP Address 

SERVER="`hostname`"

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST"))
nodes_array=($nodes)
head_node=${nodes_array[0]}
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export MASTER_PORT=12340
echo "Master IP address: " $MASTER_ADDR

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
#export CUDA_VISIBLE_DEVICES=0,1,2,3
cd workspace
#srun /home/$USER/.conda/envs/env_mnode/bin/python source_code/slurm_ddp-optimize.py
#srun nsys profile --trace cuda,osrt,nvtx,ucx \
#--capture-range cudaProfilerApi \
#--gpu-metrics-device=all \
#--gpu-metrics-frequency=5000 \
#--cuda-flush-interval=0 \
#--nic-metrics=true \
#--output reports/multinode_nvtx \
#--force-overwrite true /home/$USER/.conda/envs/env_mnode/bin/python source_code/slurm_ddp-optimize.py
#limits the viz of each devices. so the prof does not see all the devices
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
srun nsys profile --trace cuda,osrt,nvtx,ucx \
--gpu-metrics-devices=cuda-visible \
--gpu-metrics-frequency=5000 \
--cuda-flush-interval=0 \
--nic-metrics=true \
--output reports/multinode_llm_nvtx_%q{SLURM_NODEID}_%q{SLURM_PROCID} \
--force-overwrite true /home/$USER/.conda/envs/env_mnode/bin/python source_code/fsdp_tp_example.py


#srun torchrun --nproc_per_node=4 --nnodes=2 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:29500 source_code/slurm_ddp.py
#torchrun --nnodes=1 --nproc_per_node=${2:-4} --rdzv_id=101 --rdzv_endpoint="localhost:5972" ${1:-fsdp_tp_example.py}