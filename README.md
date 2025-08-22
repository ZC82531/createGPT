# createGPT
This project is a fresh implementation inspired by Andrej Karpathy's nanoGPT. It leverages the Transformer architecture and is trained on the Tiny Shakespeare Dataset.

## Dependencies
The essential dependency is PyTorch. To install it, run:
```bash
cd PROJECT_PATH
pip3 install torch
```

## Getting Started
All hyperparameters are located within `train.py`. Adjust them as you like, then start training with:

```bash
python3 train.py
```

That's all! Training typically completes in about 15 minutes on a single RTX 3090 GPU.

## Generated Texts

After training, I used the model with the best validation performance to generate some sample outputs. Here are a few examples:

```text
First Citizen:
Thou drink me, prick thee with violence. The vest,
could noise the valour tears of my veins
sheelf captain and walked on her is to be
saily: I would it be it in not ask; that
prophest he shall reported to heaven
the heavens, whose hath only she well to poss. Even
he came to for my wanton audic testrate of myself:
Whom thou have strange force thyself tears:
it demand thee, thou calls, make too princies,
fear, if Oxford thou wert through it shall ere root.
```

```text
WICKING RICHARD III:
But unfold, gear me the child, I'll revenge
They and, in threw'd, and law must surfeils;
I knock'd, he's death, which they foreget,
Cuount the sunshines yours, church ambs;
Furst thence they duty use dews, us I grustly;
And then him purposses in sometitute,
To bide me down of their earth drown bold!
O wolt, I am sugg'st husbily sound and mistrest
Slungs. Warwick! King Richard, Margius lives.
```