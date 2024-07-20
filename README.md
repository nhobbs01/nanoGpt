# nanoGpt
Code GPT in pytorch

Attention is all you need: https://arxiv.org/pdf/1706.03762

Part of the Zero-to-Hero series by Andrej Karpathy


bigram.py :
- started with embedding table. Then added positional encodings

v2.py:
- Went from bigram.py to build the full transformer from the paper, step by step.
- Runnable on cpu

v3.py:
- Same as v2.py but scaled up the hyper parameters to run on GPU.