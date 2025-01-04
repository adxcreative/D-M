# Number of GPUs per GPU worker
GPUS_PER_NODE=1
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR=localhost
# The port for communication
export MASTER_PORT=8528
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} eval.py \
  --batch_size 32 \
  --max_epochs 3000 \
  --tans_start_epoch 1000 \
  --nega_num 50 \
  --num-workers 12 \
  --lr=3e-4 \
  --wd=0.001 \
  --HM_match 1 \
  --HM_l1 5 \
  --HM_giou 5 \
  --fore_match 1 \
  --l1 4 \
  --giou 1 \
  --back_match 1 \
  --train_model_path 'save/train.pth'