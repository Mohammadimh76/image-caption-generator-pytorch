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
from torcheval.metrics import BLEUScore  # BLEU score for evaluating text generation
import torchmetrics as tm  # Additional ML metrics for PyTorch

# Progress bar for loops
import tqdm  # Displays progress bars for iterations

# FastAPI for backend server
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io

#Section: Utils
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

# Initialize FastAPI app
app = FastAPI(title="Image Caption Generator API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load the pre-trained ResNet model"""
    global model
    try:
        # Load pre-trained ResNet model
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model = model.to(device)
        model.eval()
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess the input image for the model"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def generate_caption_from_features(features: torch.Tensor) -> str:
    """Generate a caption from the image features"""
    # This is a simple example - in a real application, you would use a proper captioning model
    # For now, we'll return a placeholder caption
    return "A sample caption for the uploaded image"

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    success = load_model()
    if not success:
        raise RuntimeError("Failed to load the model")

@app.post("/generate-caption")
async def generate_caption(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Generate caption for the uploaded image"""
    try:
        # Read and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        processed_image = preprocess_image(image)
        
        # Get image features
        with torch.no_grad():
            features = model(processed_image)
        
        # Generate caption
        caption = generate_caption_from_features(features)
        
        return {
            "caption": caption,
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)
    
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)