import numpy as np
import torch
import random
import pickle
import argparse
import pathlib
from tqdm import tqdm
from implicit_maml.dataset import OmniglotTask, OmniglotFewShotDataset

np.random.seed(123)
torch.manual_seed(123)
random.seed(123)

# There are 1623 characters (for Omniglot)
train_val_permutation = list(range(1623))
random.shuffle(train_val_permutation)

parser = argparse.ArgumentParser(description='Pregenerate task definitions')
parser.add_argument('--data_dir', type=str, default='/home/aravind/data/omniglot-py/',
                    help='location of the dataset')
parser.add_argument('--task', type=str, default='Omniglot')
parser.add_argument('--num_tasks', type=int, default=20000)
parser.add_argument('--N_way', type=int, default=5, help='number of classes for few shot learning tasks')
parser.add_argument('--K_shot', type=int, default=1, help='number of instances for few shot learning tasks')
parser.add_argument('--save_dir', type=str, default='/home/aravind/data/')
args = parser.parse_args()
assert args.task == 'Omniglot'

print("Generating tasks ...... ")
task_defs = [OmniglotTask(train_val_permutation, root=args.data_dir, num_cls=args.N_way, num_inst=args.K_shot) for _ in tqdm(range(args.num_tasks))]
    
print("Saving task defs ......")
pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
save_file = args.save_dir + '/' + args.task + '_' + str(args.N_way) + '_way_' + str(args.K_shot) + '_shot.pickle'
pickle.dump(task_defs, open(save_file, 'wb'))
