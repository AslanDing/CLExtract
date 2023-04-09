import os
import numpy as np
from sklearn.model_selection import train_test_split

def loadData(name,data_dir = '../raw/binary/',mode=None):
    file = 'positive_'+name
    with open(data_dir + file + '.npy','rb') as fin:
        pos = np.load(fin)

    file = 'mask_'+name
    with open(data_dir + file + '.npy','rb') as fin: # ../.
        mask = np.load(fin)
        
    file = 'negative_'+name
    with open(data_dir + file + '.npy','rb') as fin: # ../.
        neg = np.load(fin)[:pos.shape[0]]

    mask2 = np.concatenate([mask,mask],0)
    X = np.concatenate([pos,neg],0)
    y = np.concatenate([np.ones(pos.shape[0]),np.zeros(neg.shape[0])],0)

    if mode=='positive':
        return pos, mask

    return X,y,mask2


if __name__=="__main__":
    loadData("ipheaders")
