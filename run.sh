#! /bin/bash

NUM_GPUS=$1
shift

python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} run_dee_task.py $*

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 run_dee_task.py

# ner_model = BertForBasicNER.from_pretrained(self.setting.bert_model, num_entity_labels = self.setting.num_entity_labels, gradient_checkpointing=True)