import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import ShakespeareDataset
import os
import requests
from typing import Dict, Union, Tuple


def download_data(input_file_path: str):
    # This is like going to the library to get a Shakespeare book
    # But instead of walking there, we download it from the internet
    
    # Check if we already have the book - no need to download twice!
    if not os.path.exists(input_file_path):
        print("Downloading dataset...")
        # This is the internet address where Shakespeare's works are stored
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        # Download the book and save it to our computer
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)
        print("Dataset is downloaded to {}".format(input_file_path))


def return_dataset(
        data_path: int,     # Where to find our Shakespeare book
        split: float,       # How much to use for learning vs. testing (like 0.9 = 90% learn, 10% test)
        block_size: int     # How many letters in each lesson
    ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:

    # Open our Shakespeare book and read everything
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Find all unique letters and sort them (so 'a' always comes before 'b', etc.)
    characters = sorted(list(set(text)))
    dataset_len = len(text)
    
    # Decide where to split: first 90% for learning, last 10% for testing
    train_size = int(dataset_len * split)

    # Cut the book into two parts
    train_text = text[:train_size]      # First part: for learning patterns
    test_text = text[train_size:]       # Last part: for checking if we learned correctly
    
    # Create two librarians: one for practice, one for final exams
    train_set = ShakespeareDataset(train_text, characters, block_size, train=True)   # Practice mode: random lessons
    test_set = ShakespeareDataset(test_text, characters, block_size, train=False)    # Exam mode: systematic lessons
    return train_set, test_set


if __name__ == "__main__":
    # Let's test our helper functions
    path = "./input.txt"
    split = 0.8          # Use 80% for learning, 20% for testing
    block_size = 256     # Each lesson is 256 letters long

    # Create our two librarians (practice and exam)
    train, test = return_dataset(path, split, block_size)

    # See what we got
    print("train len", len(train))    # How many practice lessons?
    print("test len", len(test))      # How many exam questions?

    # Peek at what the librarians are serving up
    print("train sample", train.text[:100])  # First 100 letters of practice material
    print("test sample", test.text[:100])    # First 100 letters of exam material