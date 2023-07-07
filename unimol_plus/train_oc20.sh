[ -z "${MASTER_PORT}" ] && MASTER_PORT=10088
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0

[ -z "${lr}" ] && lr=2e-4
[ -z "${end_lr}" ] && end_lr=1e-9
[ -z "${warmup_steps}" ] && warmup_steps=150000
[ -z "${total_steps}" ] && total_steps=1500000
[ -z "${update_freq}" ] && update_freq=1
[ -z "${seed}" ] && seed=1
[ -z "${clip_norm}" ] && clip_norm=5
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${pos_loss_weight}" ] && pos_loss_weight=12.0
[ -z "${dist_loss_weight}" ] && dist_loss_weight=5.0
[ -z "${min_pos_loss_weight}" ] && min_pos_loss_weight=1.0
[ -z "${min_dist_loss_weight}" ] && min_dist_loss_weight=1.0
[ -z "${noise}" ] && noise=0.3
[ -z "${label_prob}" ] && label_prob=0.8
[ -z "${mid_prob}" ] && mid_prob=0.1
[ -z "${mid_lower}" ] && mid_lower=0.4
[ -z "${mid_upper}" ] && mid_upper=0.6
[ -z "${ema_decay}" ] && ema_decay=0.999
[ -z "${arch}" ] && arch=unimol_plus_oc20_base


export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
echo "n_gpu per node" $n_gpu
echo "OMPI_COMM_WORLD_SIZE" $OMPI_COMM_WORLD_SIZE
echo "OMPI_COMM_WORLD_RANK" $OMPI_COMM_WORLD_RANK
echo "MASTER_IP" $MASTER_IP
echo "MASTER_PORT" $MASTER_PORT
echo "data" $1
echo "save_dir" $2
echo "warmup_step" $warmup_step
echo "total_step" $total_step
echo "update_freq" $update_freq
echo "seed" $seed
echo "valid_sets" $valid_sets

data_path=$1
save_dir=$2
lr=$3
batch_size=$4

more_args=""



more_args=$more_args" --ema-decay $ema_decay --validate-with-ema"
save_dir=$save_dir"-ema"$ema_decay

if [ -z "${train_with_valid_data}" ]
then
  echo "normal training"
else
  echo "training with additional validation data"
  more_args=$more_args" --train-with-valid-data"
  save_dir=$save_dir"-full"
fi


mkdir -p $save_dir

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

torchrun --nproc_per_node=$n_gpu --nnodes=$OMPI_COMM_WORLD_SIZE  --node_rank=$OMPI_COMM_WORLD_RANK  --master_addr=$MASTER_IP --master_port=$MASTER_PORT \
      $(which unicore-train) $data_path --user-dir ./unimol_plus \
      --num-workers 8 --ddp-backend=no_c10d \
      --task oc20 --loss unimol_plus --arch $arch \
      --train-subset train --valid-subset val_id,val_ood_ads,val_ood_cat,val_ood_both --best-checkpoint-metric loss \
      --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir $save_dir/tsb \
      --log-interval 100 --log-format simple \
      --save-interval-updates 10000 --validate-interval-updates 10000 --keep-interval-updates 10 --no-epoch-checkpoints  \
      --save-dir $save_dir \
      --batch-size $batch_size \
      --data-buffer-size 32 --fixed-validation-seed 11 --batch-size-valid $((batch_size*4)) \
      --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm $clip_norm \
      --lr $lr --end-learning-rate $end_lr --lr-scheduler polynomial_decay --power 1 \
      --warmup-updates $warmup_steps --total-num-update $total_steps --max-update $total_steps --update-freq $update_freq \
      --weight-decay $weight_decay \
      --dist-loss-weight $dist_loss_weight --pos-loss-weight $pos_loss_weight \
      --min-dist-loss-weight $min_dist_loss_weight --min-pos-loss-weight $min_pos_loss_weight \
      --label-prob $label_prob --noise-scale $noise  \
      --mid-prob $mid_prob --mid-lower $mid_lower --mid-upper $mid_upper --seed $seed $more_args