import numpy as np
import torch
import torch.nn as nn
import implicit_maml.utils as utils
import random
import time as timer
import pickle
import argparse

from tqdm import tqdm
from implicit_maml.dataset import OmniglotTask, OmniglotFewShotDataset
from implicit_maml.learner_model import Learner
from implicit_maml.learner_model import make_fc_network, make_conv_network
from implicit_maml.utils import DataLog

SEED = 12345

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# There are 1623 characters
train_val_permutation = list(range(1623))
random.shuffle(train_val_permutation)

# ===================
# hyperparameters
# ===================
parser = argparse.ArgumentParser(description='Parameters for test performance')
parser.add_argument('--task', type=str, default='Omniglot')
parser.add_argument('--data_dir', type=str, default='/home/aravind/data/omniglot-py/',
                    help='location of the dataset')
parser.add_argument('--train_set', action='store_true')  # defaults to False if nothing is provided to the argument, if --train_set is passed on, then would become true
parser.add_argument('--N_way', type=int, required=True)
parser.add_argument('--K_shot', type=int, required=True)
parser.add_argument('--num_tasks', type=int, default=600)

parser.add_argument('--load_agent', type=str, required=True)
parser.add_argument('--n_steps', type=int, default=5)
parser.add_argument('--lam', type=float, default=0.0)
parser.add_argument('--inner_lr', type=float, default=1e-2)
parser.add_argument('--inner_alg', type=str, default='gradient')
parser.add_argument('--use_gpu', type=bool, default=True)
args = parser.parse_args()
assert args.task == 'Omniglot' or args.task == 'MiniImageNet'
print(args)

print("Generating tasks ...... ")
print("Meta-training set? : ", str(args.train_set))
task_defs = [OmniglotTask(train_val_permutation, root=args.data_dir, num_cls=args.N_way, num_inst=args.K_shot, train=args.train_set) for _ in tqdm(range(args.num_tasks))]
dataset = OmniglotFewShotDataset(task_defs=task_defs, GPU=args.use_gpu)

print("Loading agent")
meta_learner = pickle.load(open(args.load_agent, 'rb'))
meta_learner.set_params(meta_learner.get_params())
fast_learner = pickle.load(open(args.load_agent, 'rb'))
fast_learner.set_params(fast_learner.get_params())
for learner in [meta_learner, fast_learner]:
    learner.inner_alg = args.inner_alg
    learner.set_inner_lr(args.inner_lr)

init_params = meta_learner.get_params()
device = 'cuda' if args.use_gpu is True else 'cpu'
lam = torch.tensor(args.lam)
lam = lam.to(device)

losses = np.zeros((args.num_tasks, 4))
accuracy = np.zeros((args.num_tasks, 2))
meta_params = meta_learner.get_params().clone()

for i in tqdm(range(args.num_tasks)):
    fast_learner.set_params(meta_params)
    task = dataset.__getitem__(i)
    vl_before = fast_learner.get_loss(task['x_val'], task['y_val'], return_numpy=True)
    if args.lam <= 0.0:
        tl = fast_learner.learn_task(task, num_steps=args.n_steps, add_regularization=False)
    else:
        tl = fast_learner.learn_task(task, num_steps=args.n_steps, add_regularization=True, w_0=meta_params, lam=args.lam)
    vl_after = fast_learner.get_loss(task['x_val'], task['y_val'], return_numpy=True)
    tacc = utils.measure_accuracy(task, fast_learner, train=True)
    vacc = utils.measure_accuracy(task, fast_learner, train=False)
    losses[i] = np.array([tl[0], vl_before, tl[-1], vl_after])
    accuracy[i][0] = tacc; accuracy[i][1] = vacc

print("Mean accuracy: on train set and val set on specified tasks (meta-train or meta-test)")  
print(np.mean(accuracy, axis=0))
print("95% confidence intervals for the mean")
print(1.96*np.std(accuracy, axis=0)/np.sqrt(args.num_tasks))  # note that accuracy is already in percentage

