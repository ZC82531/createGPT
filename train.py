import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
from typing import Dict, Union, Tuple
from model import GPT
from utils import download_data, return_dataset


def train_one_epoch(
        train_loader: torch.utils.data.DataLoader,  # Our batch of practice lessons
        model: torch.nn.Module,                     # The student (our GPT model)
        criterion: torch.nn.Module,                 # The teacher who grades the answers
        optimizer: torch.optim.Optimizer,           # The tutor who helps the student improve
        scheduler: torch.optim.lr_scheduler,        # Adjusts how fast the student learns
        device: str                                 # Whether we're using regular brain (CPU) or super brain (GPU)
    ) -> Dict[str, Union[torch.tensor, float]]:
    # This is like ONE DAY of school where the student practices lots of lessons
 
    start = time.time()
    model.train()  # Tell the student: "Time to learn!" (enables learning mode)
    losses = torch.zeros(len(train_loader))  # Keep track of how many mistakes on each lesson
    
    for i, sample in enumerate(train_loader):
        # Move the lesson to the right type of brain (CPU or GPU)
        X = sample["X"].to(device)    # The question: "To be or not to b"
        y = sample["y"].to(device)    # The right answer: "o be or not to be"
        text = sample["text"]         # Human-readable version for us to check
        
        # Student attempts to answer the question
        logits = model(X)
        # Teacher grades the answer: how far off was the student?
        loss = criterion(logits, y.view(-1,))  # Compare prediction to correct answer
        losses[i] = loss.item()  # Record the mistake level
        
        # Learning process: figure out what went wrong and fix it
        optimizer.zero_grad()  # Clear previous learning notes
        loss.backward()        # Figure out what needs to change
        optimizer.step()       # Actually make the student smarter
    
    scheduler.step()  # Adjust learning speed for tomorrow
    time_elapsed = time.time() - start
    train_info = {"loss": torch.mean(losses), "time": time_elapsed}
    return train_info


def test_one_epoch(
        test_loader: torch.utils.data.DataLoader,  # Our batch of exam questions
        model: torch.nn.Module,                    # The student taking the exam
        criterion: torch.nn.Module,                # The teacher grading the exam
        device: str                                # Which type of brain to use
    ) -> Dict[str, Union[torch.tensor, float]]:
    # This is like EXAM DAY - no learning, just testing what the student knows

    start = time.time()
    model.eval()  # Tell the student: "Exam mode - no cheating, no learning during test!"
    losses = torch.zeros(len(test_loader))  # Track mistakes on each exam question
    
    # No learning allowed during the exam!
    with torch.inference_mode():
        for i, sample in enumerate(test_loader):
            # Give the student the exam questions
            X = sample["X"].to(device)
            y = sample["y"].to(device)
            text = sample["text"]
            # Student answers (but can't learn from mistakes yet)
            logits = model(X)
            loss = criterion(logits, y.view(-1,))
            losses[i] = loss.item()
    
    time_elapsed = time.time() - start
    test_info = {"loss": torch.mean(losses), "time": time_elapsed}
    return test_info


def generate_text(
        model: torch.nn.Module,  # Our trained student who learned Shakespeare
        device: str,             # Which brain to use
        num_tokens: int          # How many letters to write
    ):
    # This is like asking the student: "Write me a Shakespeare poem!"
    # Start with just one letter and let the student continue
    idx = torch.zeros((1,1), dtype=torch.long).to(device)
    # Generate text and convert numbers back to readable letters
    print(train_set.decoder(model.generate(idx, num_tokens)[0].tolist()))


if __name__ == "__main__":
    # SCHOOL CONFIGURATION - These are like the rules for our Shakespeare school
    data_path = "./input.txt"        # Where we keep the Shakespeare textbook
    load_path = None                 # Should we start with a student who already knows something? (None = fresh start)
    epochs = 50                      # How many days of school (50 full days of learning)
    block_size = 256                 # How many letters in each lesson (256 letters at a time)
    split = 0.9                      # How much text for practice vs. testing (90% practice, 10% testing)
    batch_size = 64                  # How many lessons per study session (64 different text chunks)
    initial_lr = 3e-4                # How eager the student is to learn at first (0.0003 = pretty eager)
    min_lr = 1e-4                    # Minimum learning eagerness (never get too lazy)
    evaluate_every = 10              # How often to give exams (every 10 days)
    n_embed = 384                    # How detailed each letter's "personality card" is (384 traits)
    num_heads = 6                    # How many different perspectives to consider (6 different viewpoints)
    n_layers = 6                     # How many processing stations in our factory (6 layers deep)
    device_id = 0                    # Which GPU to use (if we have multiple super-brains)
    checkpoint_dir = "./results/"    # Where to save our best students (model saves)

    # Get the Shakespeare textbook (download if we don't have it)
    download_data(data_path)

    # Create a folder to save our best students
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Prepare the lessons: split Shakespeare into practice and exam material
    train_set, test_set = return_dataset(data_path, split, block_size)

    # PREPARE THE CLASSROOM
    # Create lesson organizers that group lessons into batches
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # Choose which brain to use: super brain (GPU) if available, regular brain (CPU) otherwise
    device = torch.device('cuda:{}'.format(device_id) if torch.cuda.is_available() else 'cpu')

    # CREATE OUR STUDENT
    num_chars = len(train_set.characters)  # How many different letters our student needs to learn
    model = GPT(num_chars, block_size, n_embed, num_heads, n_layers)
    # If we have a pre-trained student, load their knowledge
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))
    model = model.to(device)  # Put the student's brain on the right processor

    # SET UP THE TEACHING SYSTEM
    criterion = nn.CrossEntropyLoss()  # The teacher who grades answers (measures how wrong predictions are)
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)  # The tutor who helps fix mistakes
    
    # Learning speed controller: start eager, gradually become more careful
    lambda_func = lambda epoch: max(0.99 ** epoch, min_lr / initial_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)

    # SCHOOL BEGINS! 
    # This is like 50 days of Shakespeare school
    best_val_loss = 1e5  # Keep track of the student's best exam score ever
    for e in range(epochs):
        # One day of practice lessons
        train_info = train_one_epoch(train_dataloader, model, criterion, optimizer, scheduler, device)
        print("At epoch: {}, train loss: {:.2f}, in {:.2f} seconds".format(e+1, train_info["loss"], train_info["time"]))
        
        # Every 10 days, give an exam to see how the student is doing
        if (e+1) % evaluate_every == 0:
            test_info = test_one_epoch(test_dataloader, model, criterion, device)
            print("\nAt epoch: {}, test loss: {:.2f}, in {:.2f} seconds\n".format(e+1, test_info["loss"], test_info["time"]))
            # If this is the best exam score ever, save this student!
            if best_val_loss > test_info["loss"]:
                torch.save(model.state_dict(), checkpoint_dir + "model_epoch_{}_loss_{:.2f}.pt".format(e, test_info["loss"]))
                best_val_loss = test_info["loss"]

    # GRADUATION DAY!
    # Ask our newly trained student to write some Shakespeare
    generate_text(model, device, 500)
