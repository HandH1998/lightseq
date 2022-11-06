# Copyright 2021 The LightSeq Team
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

THIS_DIR=$(dirname $(readlink -f $0))
# export http_proxy=""
# export https_proxy=""

# You can use multiple NICs in NCCL communication.
# E.g., if every machine has 4 NICs: eth0, eth1, eth2, eth3, you can use the following command.
# export NCCL_SOCKET_IFNAME=eth0,eth1,eth2,eth3

# Set your environment variables according to your training environment,
# for details, please refer to https://pytorch.org/docs/1.10/distributed.html#launch-utility
python3 -m torch.distributed.launch --nproc_per_node=$WORKER_GPU_NUM \
  --nnodes=$WORKER_NUM --node_rank=$WORKER_ID --master_addr=$WORKER_0_HOST \
  --master_port=$WORKER_0_PORT \
  $THIS_DIR/run_gcq_translation.py \
  --model_name_or_path Helsinki-NLP/opus-mt-en-de \
  --do_train \
  --do_eval \
  --source_lang en \
  --target_lang de \
  --dataset_name wmt16 \
  --dataset_config_name de-en \
  --max_source_length 512 \
  --output_dir /tmp/tst-translation \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 25 \
  --predict_with_generate \
  --fp16 \
  --seed 1234 \
  --logging_steps 100 \
  --logging_dir $THIS_DIR/log/gcq \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --save_strategy epoch \
  --enable_GCQ 

