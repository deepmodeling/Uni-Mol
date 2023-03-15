[ -z "${MASTER_PORT}" ] && MASTER_PORT=10087
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=0


[ -z "${arch}" ] && arch=unimol_plus_base

mkdir -p $results_path

python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $MASTER_PORT --nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_IP \
       ./inference.py --user-dir ./unimol_plus/ $data_path --valid-subset $1 \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size 128 \
       --task unimol_plus --loss unimol_plus --arch $arch  \
       --path $weight_path \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 50 --log-format simple --label-prob 0.0 