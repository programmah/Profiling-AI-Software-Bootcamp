#!/bin/bash
# $SLURM_PROCID, $PMI_RANK (MPICH), $OMPI_COMM_WORLD_RANK, etc.
if [ $SLURM_LOCALID -eq 0 ]; then
  nsys profile --enable=network_interface,-i10000 "$@"
  #            --nic-metrics=all
  # -t mpi -e NSYS_MPI_STORE_TEAMS_PER_RANK=1 
else
  nsys profile "$@"
fi
