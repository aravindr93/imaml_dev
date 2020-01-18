"""
- We will have two classes: task defs and task dataset
- For sinusoid, they are the same.
- The two can be different for large scale tasks. For omniglot, task def contains the location of images, and
  task dataset contains the logic to load and process images to give to the learner
- Each task is a dict with the following keys: x_train, y_train, x_val, y_val, x_all, y_all
  x_all and y_all are simply concatenations of train and val
"""

import os
import random
import numpy as np
import torch
import pickle
import torch.nn as nn
import matplotlib.pyplot as plt
import implicit_maml.utils as utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.ndimage import rotate

DATA_DIR = '/home/aravind/data/omniglot-py/'

class SinusoidDataset(Dataset):
    def __init__(self, num_tasks=100, train_inst=10, val_inst=10, GPU=False, float16=False):
        """
        :param num_tasks: number of tasks in this dataset (create seperate dataset for meta-testing phase)
        :param train_inst: K in K-shot learning
        :param val_inst: ideally K in K-shot learning, but can be different
        :param GPU: bool, True to run on GPU and False to run on CPU
        """
        self.num_tasks = num_tasks
        self.ntrain, self.nval = train_inst, val_inst
        self.amp_range = [0.1, 5.0]
        self.phase_range = [0.0, np.pi]
        self.input_range = [-5.0, 5.0]
        self.use_gpu = GPU
        self.float16 = float16

        # generate the tasks and store in memory (small task, so this is fine)
        self.task_data = self.generate_tasks()

        super(SinusoidDataset, self).__init__()

    def generate_tasks(self, num_tasks=None):
        num_tasks = self.num_tasks if num_tasks == None else num_tasks
        generated_tasks = []
        for i in range(num_tasks):
            amp = np.random.uniform(low=self.amp_range[0], high=self.amp_range[1])
            phase = np.random.uniform(low=self.phase_range[0], high=self.phase_range[1])
            x_train = np.random.uniform(low=self.input_range[0], high=self.input_range[1], size=self.ntrain).reshape(-1,
                                                                                                                     1)
            y_train = self.sine_function(x_train, amp, phase).reshape(-1, 1)
            x_val = np.random.uniform(low=self.input_range[0], high=self.input_range[1], size=self.nval).reshape(-1, 1)
            y_val = self.sine_function(x_val, amp, phase).reshape(-1, 1)
            x_all = np.concatenate([x_train, x_val])
            y_all = np.concatenate([y_train, y_val])
            if self.float16:
                task = dict(amp=amp,
                            phase=phase,
                            x_train=utils.to_device(x_train, self.use_gpu).half(),
                            y_train=utils.to_device(y_train, self.use_gpu).half(),
                            x_val=utils.to_device(x_val, self.use_gpu).half(),
                            y_val=utils.to_device(y_val, self.use_gpu).half(),
                            x_all=utils.to_device(x_all, self.use_gpu).half(),
                            y_all=utils.to_device(y_all, self.use_gpu).half(),
                            )
            else:
                task = dict(amp=amp,
                            phase=phase,
                            x_train=utils.to_device(x_train, self.use_gpu),
                            y_train=utils.to_device(y_train, self.use_gpu),
                            x_val=utils.to_device(x_val, self.use_gpu),
                            y_val=utils.to_device(y_val, self.use_gpu),
                            x_all=utils.to_device(x_all, self.use_gpu),
                            y_all=utils.to_device(y_all, self.use_gpu),
                            )
            generated_tasks.append(task)
        return generated_tasks

    def sine_function(self, x, amp, phase):
        """
        y = amp * sin(x + phase)
        """
        return amp * np.sin(x + phase)

    def __len__(self):
        """
        Should return the number of elements (i.e. tasks) in the dataset
        To be used with data loader
        """
        return self.num_tasks

    def __getitem__(self, idx):
        """
        Should return a task
        To be used with data loader
        """
        return self.task_data[idx]


class OmniglotTask(object):
    """
    Create the task definition for N-way k-shot learning with Omniglot dataset
    Assumption: number of train and val instances are same (easy to lift in the future)
    """
    def __init__(self, train_val_permutation, root=DATA_DIR, num_cls=5, num_inst=1, train=True):
        """
        :param train_val_permutation: permutation of the 1623 characters, first 1200 are for train, rest for val
        :param root: location of the dataset
        :param num_cls: number of classes in task instance (N-way)
        :param num_inst: number of instances per class (k-shot)
        :param train: bool, True if meta-training phase and False if test/deployment phase
        """
        # different sampling stratergy
        # 1200 classes for meta-train phase and rest for test phase
        self.root1 = os.path.join(root, 'images_background')
        self.root2 = os.path.join(root, 'images_evaluation')
        self.num_cls = num_cls
        self.num_inst = num_inst
        # Sample num_cls characters and num_inst instances of each
        languages1 = os.listdir(self.root1)
        languages2 = os.listdir(self.root2)
        languages1.sort()
        languages2.sort()
        chars = []
        for l in languages1:
            chars += [os.path.join(self.root1, l, x) for x in os.listdir(os.path.join(self.root1, l))]
        for l in languages2:
            chars += [os.path.join(self.root2, l, x) for x in os.listdir(os.path.join(self.root2, l))]
        chars = np.array(chars)[train_val_permutation]
        chars = chars[:1200] if train else chars[1200:]
        random.shuffle(chars)
        classes = chars[:num_cls]
        labels = np.array(range(len(classes)))
        labels = dict(zip(classes, labels))
        instances = dict()
        # Now sample from the chosen classes to create class-balanced train and val sets
        self.train_ids = []
        self.val_ids = []
        for c in classes:
            # First get all isntances of that class
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            instances[c] = random.sample(temp, len(temp))
            # Sample num_inst instances randomly each for train and val
            self.train_ids += instances[c][:num_inst]
            self.val_ids += instances[c][num_inst:num_inst * 2]
        # Keep instances separated by class for class-balanced mini-batches
        self.train_labels = [labels[self.get_class(x)] for x in self.train_ids]
        self.val_labels = [labels[self.get_class(x)] for x in self.val_ids]

    def get_class(self, instance):
        return '/' + os.path.join(*instance.split('/')[:-1])


class OmniglotFewShotDataset(Dataset):
    """`Omniglot <https://github.com/brendenlake/omniglot>`_ Dataset.
    Assumption:
        The dataset has been downloaded and placed in task root (the task defs cannot be generated otherwise)
    Args:
        task_defs: task definitions for the few-shot learning problem
                    a list with each element being of type OmniglotTask
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        GPU(bool, optional): True to run on GPU and False to run on CPU
    """

    def __init__(self, task_defs, transform=None, target_transform=None, GPU=False):
        self.transform = transform
        self.target_transform = target_transform
        self.task_defs = task_defs
        self.use_gpu = GPU
        super(OmniglotFewShotDataset, self).__init__()

    def __len__(self):
        """
        Should return the number of elements (i.e. tasks) in the dataset
        To be used with data loader
        """
        return len(self.task_defs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            task dictionary with keys x_train, y_train, x_val, y_val, all draw from the task of given index
        """
        assert index < self.__len__()
        task_def = self.task_defs[index]
        x_train = []
        y_train = []
        x_val   = []
        y_val   = []

        rotations = np.random.choice([0., 90., 180., 270.], size=(task_def.num_cls,), replace=True)

        # training instances
        for idx, file_name in enumerate(task_def.train_ids):
            image_path = file_name
            image = Image.open(image_path, mode='r').convert('L')
            image = image.resize((28,28), resample=Image.LANCZOS)
            image = np.array(image, dtype=np.float32)
            image = rotate(image, rotations[idx // task_def.num_inst])
            x_train.append(image.reshape(1, 28, 28)/255.0)
            y_train.append(task_def.train_labels[idx])

        # validation instances
        for idx, file_name in enumerate(task_def.val_ids):
            image_path = file_name
            image = Image.open(image_path, mode='r').convert('L')
            image = image.resize((28,28), resample=Image.LANCZOS)
            image = np.array(image, dtype=np.float32)
            image = rotate(image, rotations[idx // task_def.num_inst])
            x_val.append(image.reshape(1, 28, 28)/255.0)
            y_val.append(task_def.val_labels[idx])


        # base transforms
        x_train = utils.to_device(np.array(x_train), self.use_gpu)
        y_train = utils.to_device(np.array(y_train), self.use_gpu)
        x_val   = utils.to_device(np.array(x_val), self.use_gpu)
        y_val   = utils.to_device(np.array(y_val), self.use_gpu)

        if self.transform:
            x_train = self.transform(x_train)
            x_val   = self.transform(x_val)

        if self.target_transform:
            y_train = self.target_transform(y_train)
            y_val   = self.target_transform(y_val)

        # cross entropy expects targets to be of type LongTensor
        task = dict(task_def=task_def,
                    x_train=x_train,
                    y_train=y_train.long(),
                    x_val=x_val,
                    y_val=y_val.long())

        return task

    
# TODO(Aravind): Add mini-imagenet to this version of code