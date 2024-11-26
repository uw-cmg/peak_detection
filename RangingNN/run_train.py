import torch
import os,time
from RangingNN.trainer import BaseTrainer
from pathlib import Path
import yaml
print(torch.cuda.is_available(),torch.__version__)

if __name__ == '__main__':
    start = time.time()
    yaml_dict = yaml.safe_load(Path("/srv/home/jwei74/APT_ML/ranging/current_all_args.yaml").read_text())

    trainer = BaseTrainer(cfg=yaml_dict)
    start = time.time()
    trainer.train()
    print('Took: ', time.time() - start)

