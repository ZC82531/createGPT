import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from typing import Dict, Union, Tuple


class ShakespeareDataset(torch.utils.data.Dataset):
    # This is like a LIBRARIAN that cuts up Shakespeare into bite-sized pieces for learning
    # Instead of reading the whole book at once, we read small chunks and learn patterns
    def __init__(
            self, 
            text: str,           # The entire Shakespeare text as one giant string
            characters: int,     # All the unique letters/symbols we found (like ['a', 'b', 'c', ' ', '!'])
            block_size: int,     # How many letters to include in each "lesson" (like 256)
            train: bool=True     # Are we making lessons for practice (True) or testing (False)?
        ):
        super(ShakespeareDataset, self).__init__()
        self.text = text
        self.characters = characters
        
        # Create a "translation dictionary" between letters and numbers
        # Because computers prefer numbers to letters
        str_to_int_dict = {s:i for i,s in enumerate(self.characters)}  # 'a'->0, 'b'->1, etc.
        int_to_str_dict = {i:s for i,s in enumerate(self.characters)}  # 0->'a', 1->'b', etc.
        
        # Create "translator functions"
        self.encoder = lambda s: [str_to_int_dict[c] for c in s]       # Turn "hello" into [7,4,11,11,14]
        self.decoder = lambda l: ''.join([int_to_str_dict[i] for i in l])  # Turn [7,4,11,11,14] back to "hello"
        
        # Convert ALL of Shakespeare into a long list of numbers
        self.data = torch.tensor(self.encoder(self.text), dtype=torch.long)
        self.block_size = block_size
        self.train = train

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        # This is like asking the librarian: "Give me lesson #47"
        # Each lesson is: "Here's some text, now predict what comes next"
        
        idx = index
        if self.train:
            # For practice: pick random spots in Shakespeare (more variety = better learning)
            idx = torch.randint(len(self.data) - self.block_size, size=(1,))
        
        # Create one lesson: input text + what should come next
        X = self.data[idx:idx+self.block_size]           # Input: "To be or not to b"
        y = self.data[idx+1:idx+self.block_size+1]       # Target: "o be or not to be" (shifted by 1)
        text = self.text[idx:idx+self.block_size]        # Keep original text for humans to read
        
        # Package it up like a textbook lesson
        sample = {"X": X, "y": y, "text": text}
        return sample

    def __len__(self) -> int:
        # How many lessons can we create?
        if self.train:
            # For practice: let's say we can make 5000 different random lessons
            return 5000
        # For testing: use every possible position in the text exactly once
        return len(self.data) - self.block_size



if __name__ == "__main__":
    # Let's test our librarian with some sample text
    pth = "input.txt"
    with open(pth, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Find all unique letters in Shakespeare (probably around 65 different ones)
    characters = sorted(list(set(text)))
    block_size = 256  # Each lesson will be 256 letters long
    
    # Create our librarian (in testing mode so we get consistent results)
    dataset = ShakespeareDataset(text, characters, block_size, train=False)
    
    # Let's see what our librarian created
    print("len", len(dataset))                    # How many lessons total?
    print("sample", dataset[0]["X"].shape)       # What does the input look like?
    print("sample", dataset[0]["y"].shape)       # What does the target look like?
    print("sample", len(dataset[0]["text"]))     # How long is each text sample?
    print("sample\n", dataset[0]["text"][:150])  # Show me the first 150 characters