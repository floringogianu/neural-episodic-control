# Neural Episodic Control

Work in progress implementation of [Neural Episodic
Control](https://arxiv.org/abs/1703.01988). Currently learning Catcher but it
needs heavy tweaking and more testing because apparently frequent updates to
the feature extractor are harming the learning. Right now it works more like
the original _Model-Free Episodic Control_.

## Training

Just run `python main.py -cf catch_dev`.

## To do:

[] Implement a proper priority queue for the DND `pop` function.
[] More performance profiling.
[] Figure out if a more efficient KD-tree can be implemented.
[] Add support for Atari games.
  [] Atari feature extractor.
  [] Frame preprocessing.
