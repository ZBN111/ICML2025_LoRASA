#!/bin/sh

export LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

env="StarCraft2"
map="1c3s5z"
algo="rmappo"
exp="1c3s5z"
group_id='1c3s5z'
seed_begin=1
seed_end=5
gpu=0

# Algorithm ##############################################################################
# 0 for mappo, 1 for a2po
use_mappo=0
# ppo clip param
clip_param=0.2
# (for a2po only) clip param for other agents. Should be set to 1/2 clip_param
others_clip_param=0.1
# ppo epoch 
ppo_epoch=5
# actor lr
lr=5e-4
# critic lr (same as actor)
critic_lr=5e-4
# entropy coefficient
entropy_coef=0.01


# Parameter Sharing ######################################################################
# parameter sharing model types:
#               actor_ps_mode   critic_ps_mode
# PS        :   ps              ps
# NPS       :   nps             nps
# MTL       :   mtl             ps 
# SePS      :   seps            ps
# PS+LoRA   :   lora            ps
# SePS+LoRA :   sepslora        ps
actor_ps_mode='ps'  
critic_ps_mode='ps'  


# LoRA ###################################################################################
# lora settings, only needed if using lora/sepslora
r=8                 # rank of lora     
lora_alpha=8        # default =r, i.e. scaling=1. scaling = lora_alpha/r
# controls which of the 7 layers to apply lora, 1 for enabled, 0 means diabled. recommend '1 1 1 1 1 1 1'
lora_config='1 1 1 1 1 1 1'     


# Other ##################################################################################
# number of time steps
num_env_steps=12000000
# number of steps from which you wish the plot to start. 
# Useful for lora/sepslora, where set setp_shift to the number of steps from which lora is fine tuned
step_shift=0

# the dir from which modal checkpoints are loaded. If not lora fine tuning, set to None
# If lora/sepslora, e.g. model_dir='path to checkpoints for lora'
model_dir=None  

# save model checkpoints every n episodes
save_interval=31   

# number of rollout threads
n_t=10

project_name='1c3s5z'

# wandb user name if use wandb
wandb_usr='username'

if [ $# -ge 1 ]
then
    gpu=$1
fi

if [ $# -ge 2 ]
then
    n_t=$2
fi

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed from ${seed_begin} to ${seed_end}"
for seed in `seq ${seed_begin} ${seed_end}`;
do
    echo "seed is ${seed} use gpu ${gpu}"
    set -x
    CUDA_VISIBLE_DEVICES=${gpu} python3 onpolicy/scripts/train/train_smac.py --wandb_usr ${wandb_usr} --env_name ${env} --algorithm_name ${algo} \
     --experiment_name ${exp} --map_name ${map} --seed ${seed} \
     --n_rollout_threads ${n_t} --episode_length $((3200 / $n_t)) --use_eval True --n_run 1 --use_aim False \
     --aim_repo .aim_smac --ppo_epoch ${ppo_epoch} --loop_order ppo \
     --actor_ps_mode ${actor_ps_mode} --critic_ps_mode ${critic_ps_mode} \
     --lora_config ${lora_config} --r ${r} --lora_alpha ${lora_alpha} \
     --model_dir ${model_dir} --project_name ${project_name} \
     --num_env_steps ${num_env_steps} --step_shift ${step_shift} --save_interval ${save_interval} \
     --use_mappo ${use_mappo} --clip_param ${clip_param} --entropy_coef ${entropy_coef} \
     --group_id ${group_id} --others_clip_param ${others_clip_param} --lr ${lr} --critic_lr ${critic_lr}
done