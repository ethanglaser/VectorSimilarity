import pandas as pd
from pickle import load
import numpy as np
TRAIN_EMBED_CACHE = '/home/omer-lerman/Code/VectorNormTest/Python/train_embed.pkl'
with open(TRAIN_EMBED_CACHE, 'rb') as f:
    emb_train = load(f)
print(emb_train.shape)
np.savetxt('data_set_train.dat',emb_train)