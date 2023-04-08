export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2



date

num_rounds=1000
batch_size=10
num_epochs=1
clients_per_round=100
lr=0.1
lr_decay=2
decay_lr_every=200
reg=0.0
seed=10
aggregation='mean'
start_finetune_rounds=1000
q=-2


model="erm_cnn_log_reg"

dataset="femnist"

outf="outputs/exp/femnist/nn_vanilla_fl/"
logf="outputs/exp/femnist/nn_vanilla_fl/logs"

main_args=" -dataset ${dataset} -model ${model} "
options_basic=" --num-rounds ${num_rounds} -lr ${lr} --personalized --lr-decay ${lr_decay} --decay-lr-every ${decay_lr_every} --eval-every 5 --num_epochs ${num_epochs} --full_record True --gpu 2 --aggregation ${aggregation} --start_finetune_rounds ${start_finetune_rounds}"



for num_mali_devices in 0
do
options=" ${options_basic}  --clients-per-round ${clients_per_round} --seed $seed --num_mali_devices ${num_mali_devices}"

time python3 main.py ${main_args} $options  -reg_param $reg --output_summary_file ${outf}${dataset}_${model}_${reg}_ERM_${seed}_${num_mali_devices}_mtl_flip_new_ft_${start_finetune_rounds} 

done

date