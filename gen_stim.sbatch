#!/bin/bash 
#
#################
#gpu stuff
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --qos gpu
#################
#job name
#SBATCH --job-name=gen_tony_stim_vgg
#################
#output file name  
#SBATCH --output=gen_stim.out
#################
# a file for errors from the job
#SBATCH --error=gen_stim.err
#################
#time you think you need; default is one hour
#in minutes in this case
#SBATCH --time=12:00:00
#################
#quality of service; think of it as job priority
#################
#number of nodes you are requesting
#--nodes=1
#################
#memory per node; default is 4000 MB per CPU
# --mem=8000
#you could use --mem-per-cpu; they mean what we are calling cores
#################
#tasks to run per node; a "task" is usually mapped to a MPI processes.
# for local parallelism (OpenMP or threads), use "--ntasks-per-node=1 --cpus-per-tasks=16" instead
#--cpus-per-task=10
#################
ml load python/2.7.5
ml load caffe
