# Import necessary libraries for image processing, deep learning, and NLP tasks
# This script sets up the essential libraries needed for deep learning models involving NLP and image processing tasks.

import os  # OS module for file and directory operations
from collections import defaultdict  # Dictionary with default values
from typing import Any, Callable, Dict, List, Optional, Tuple  # Type hints for better readability

# Scientific computing and visualization libraries
import numpy as np  # Numerical computing library
import matplotlib.pyplot as plt  # Data visualization library

# Image processing
from PIL import Image  # Library for handling and manipulating images

# Natural Language Processing (NLP)
import torchtext  # NLP utilities for PyTorch
from torchtext.data.utils import get_tokenizer  # Tokenizer for text processing
from torchtext.vocab import build_vocab_from_iterator  # Vocabulary builder for NLP models

# Computer Vision
import torchvision  # Library for handling image datasets and transformations
from torchvision.datasets import VisionDataset  # Base class for vision datasets
from torchvision import transforms  # Preprocessing transformations for images
from torchvision.models import resnet50, ResNet50_Weights  # Pre-trained ResNet-50 model

# PyTorch core libraries
import torch  # PyTorch framework for deep learning
from torch import nn  # Neural network modules
from torch.utils.data import DataLoader, Dataset, random_split  # Data handling utilities
from torch.nn.utils.rnn import pad_sequence  # Padding sequences for NLP tasks

# Optimization and loss functions
from torch import optim  # Optimization algorithms
from torch.nn import functional as F  # Common loss functions and activation functions

# Evaluation metrics
from torchmetrics import BLEUScore  # BLEU score for evaluating text generation
import torchmetrics as tm  # Additional ML metrics for PyTorch

# Progress bar for loops
import tqdm  # Displays progress bars for iterations

# FastAPI for backend server
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io

# Set device to CPU
#device = torch.device('cpu')
#torch.set_num_threads(4)  # Adjust number of CPU threads as needed


#Section --> Utils
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def num_trainable_params(model):
  nums = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
  return nums

def set_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      # torch.cuda.manual_seed_all(seed)

      # torch.backends.cudnn.deterministic = True
      # torch.backends.cudnn.benchmark = False


#Section --> Arguments
seed = 8
batch_size = 128
wandb_enable = False

if wandb_enable:
    wandb_arg_name = input('Please input the WandB argument (run) name:')
    print(wandb_arg_name)


#Section --> Change the font size of the output cells
from IPython.display import HTML
#shell = get_ipython()

#def adjust_font_size():
#  display(HTML('''<style>
#    body {
#      font-size: 24px;
#    }
#  '''))

#if adjust_font_size not in shell.events.callbacks['pre_execute']:
#  shell.events.register('pre_execute', adjust_font_size)


#Section --> Custom dataset
class Flickr8k(VisionDataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        ann_file: str,
        split_file: str,
        train: bool,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.ann_file = os.path.expanduser(ann_file)
        self.train = train

        # Read {train/dev/test} files
        with open(split_file) as f:
            self.split_samples = f.read().strip().split("\n")

        # Read annotations and store in a dict
        self.ids, self.captions = [], []
        with open(self.ann_file) as fh:
            for line in fh:
                img_id, caption = line.strip().split("\t")
                if img_id[:-2] in self.split_samples:
                    self.ids.append(img_id[:-2])
                    self.captions.append(caption)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index]

        # Image
        filename = os.path.join(self.root, img_id)
        img_raw = Image.open(filename).convert("RGB")
        if self.transform is not None:
            img = self.transform(img_raw)

        # Captions
        caption = self.captions[index]
        if self.target_transform is not None:
            target = self.target_transform(caption)

        if self.train:
            return img, target
        else:
          return img, img_raw, caption

    def __len__(self) -> int:
        return len(self.ids)
    
class CaptionTransform:

    def __init__(self, caption_file):
        captions = self._load_captions(caption_file)

        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, captions),
                                               specials=['<pad>', '<unk>', '<sos>', '<eos>'])
        self.vocab.set_default_index(self.vocab['<unk>'])

        torch.save(self.vocab, 'vocab.pt')

    def __call__(self, caption):
        indices = self.vocab(self.tokenizer(caption))
        indices = self.vocab(['<sos>']) + indices + self.vocab(['<eos>'])
        target = torch.LongTensor(indices)
        return target

    def __repr__(self):
        return f"""CaptionTransform([
          _load_captions(),
          toknizer('basic_english'),
          vocab(vocab_size={len(self.vocab)})
          ])
          """

    def _load_captions(self, caption_file):
        captions = []
        with open(caption_file) as f:
            for line in f:
                _, caption = line.strip().split("\t")
                captions.append(caption)
        return captions
    
caption_transform = CaptionTransform('D:/EmKay/data/Flickr8k.token.txt')
print(caption_transform)
print(len(caption_transform.vocab))
caption_transform('how are you')

transform = transforms.Compose([transforms.ToTensor()])

dataset = Flickr8k('D:/EmKay/data/Flickr8k_Dataset',
                   'D:/EmKay/data/Flickr8k.token.txt',
                   'D:/EmKay/data/Flickr_8k.trainImages.txt',
                   train=True,
                   transform=transform,
                   target_transform=caption_transform)

print(dataset)

img, caption = dataset[50]
print(caption)
print(img)

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

root = 'D:/EmKay/data/Flickr8k_Dataset'
ann_file = 'D:/EmKay/data/Flickr8k.token.txt'
split_file = lambda phase: f'D:/EmKay/data/Flickr_8k.{phase}Images.txt'

caption_transform = CaptionTransform(ann_file)

train_set = Flickr8k(root, ann_file, split_file('train'), True, train_transform, caption_transform)
valid_set = Flickr8k(root, ann_file, split_file('dev'), True, eval_transform, caption_transform)
test_set = Flickr8k(root, ann_file, split_file('test'), False, eval_transform, caption_transform)

print(len(train_set), len(valid_set), len(test_set))


# Section --> DataLoader
def collate_fn(batch):
  if len(batch[0]) == 2:
      x_batch, y_batch = zip(*batch)
      x_batch = torch.stack(x_batch)
      y_batch = pad_sequence(y_batch, batch_first=True, padding_value=caption_transform.vocab['<pad>'])
      return x_batch, y_batch
  else:
    x_batch, x_raw, captions = zip(*batch)
    x_batch = torch.stack(x_batch)
    return x_batch, x_raw, captions
  
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_set, batch_size=batch_size*2, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=batch_size*2, collate_fn=collate_fn)

x_batch, y_batch = next(iter(train_loader))
print(x_batch.shape, y_batch.shape)



