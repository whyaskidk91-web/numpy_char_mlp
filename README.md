# numpy_char_mlp

## Creation process:
Created around early/mid October 2026.
Created only with numpy and only by hand, and a lot of trial and error to learn how numpy works.

## The model:

The base structure in both projects is a MLP, with He initialization, Adam,embeddings,cosine learn rate,loss and accuracy.

## Names:

This is an MLP that creates new names, one letter at a time. With a context of 7 letters and predicts the next one. It learns on 32k names.

|          | train | test  |
|----------|-------|-------|
| loss     | 2.11  | 2.10  |
| accuracy | 35.0% | 33.8% |

The loss fell while training from 2.44 to 2.09 in 30000 steps.
The gap between train loss and test loss is very small, that shows the MLP doesnt overfit, but learned the names structure.

Generated names (`•` = already in the training data, `=` = new):
```
melynnla      =
emaura        =
hailoni       =
yuss          =
crozlyn       =
vainah        =
maylin        •
mahbella      =
javya         =
faly          =
```
9 of 10 names are new inventions.

There is as well the option to type beginnings of a name and the MLP completes the rest how it thinks it should sound.

## Shakespeare

The same MLP structure predicts here a letter based on a 16 context window, this time the data is text based: 1million character long file of shakespeare plays.(10% used for test samples)

|          | train | test  |
|----------|-------|-------|
| loss     | 1.91  | 2.04  |
| accuracy | 39.9% | 39.9% |

The loss while training fell from 2.77 to 1.96 in 30000 steps.
The gap between train loss and test loss is a little bit noticeable, that shows the MLP does overfit a bit, though the loss results could partly depend on the fact that test samples use the last 10% of the text with different plays and characters.

Generated text (300 characters, starting from an empty window):
```
WERIT:
Be
Keech
Fhost him I me have noth seels is your,
Pasly mabounfroom!
Tark man
there now all powes no in knook in fifor, be hard.

DATCA:
Thp been to waste pow;
Nat's nous! go it noshs, my prirst dente and a mys's,
Which to my gard where.
Ily I' fuincior.

Furctood reye shen.

WIGCIO:
Aid dey,
```
The MLP learned the play structure (character names in capitals with a colon,
short lines, punctuation) and common short words ("him","your",
"there","which","have"), but most words are still invented.
Partly because a context window of 16 cant really grasp a bigger context.

## Limits and next steps:

The biggest problem is the context window and the computing time thats why i created in the next project a transformer in google colab.

## How to run:
```
pip install numpy
cd names && python names_AI_V4.py
cd ../shakespeare && python shakespeare_ANN_V3.py
```

## Data:
- `names.txt`: ~32k names from Andrej Karpathy's [makemore](https://github.com/karpathy/makemore)
- `shakespeare.txt`: Tiny Shakespeare from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn)

## Inspired:
Based on the idea from Bengio et al. 2003.

