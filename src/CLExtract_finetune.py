import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset,TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np
import random
import matplotlib.pyplot as plt
from util.tools import npcutout,npflip, torchflip,batch_flip,batch_bit_aug
from data.loadData import loadData
from util.tools import accConfusion,binariz,CustomTensorDataset

import torch.optim as optim

cuda1 = torch.device('cuda:0')

torch.manual_seed(1)

name = 'bbheader_gseheader'
# name = 'gseheader'
# name = 'ipheaders'

X,y, mask = loadData(name)

X_mask = np.concatenate([X,mask],-1)

X_mask_train, X_mask_test, y_train, y_test = train_test_split(X_mask, y, test_size=0.33, random_state=42)

dataset_mask_train = TensorDataset(torch.tensor(X_mask_train).type(torch.float32).to(cuda1), torch.tensor(y_train).type(torch.float32).to(cuda1))
dataset_mask_test = TensorDataset(torch.tensor(X_mask_test).type(torch.float32).to(cuda1), torch.tensor(y_test).type(torch.float32).to(cuda1))

train_mask_loader = DataLoader(dataset_mask_train, batch_size= 1024,shuffle=True)
test_mask_loader = DataLoader(dataset_mask_test, batch_size= 1024,shuffle=False)

thres = 0.5

def evl(model, eval_loader):
    model.eval()
    y_test_batch = []
    y_pred_test = []
    with torch.no_grad():
        for xm, y in eval_loader:
            leng = int(xm.shape[1] / 2)
            x = (xm[:, :leng]).to(cuda1)
            masks = (xm[:, leng:]).to(cuda1)
            y_pred_test.append(model((x, masks)).cpu())
            y_test_batch.append(y)
    y_test_batch = torch.concat(y_test_batch, 0)
    y_pred_test = torch.concat(y_pred_test, 0)
    y_pred_test = [binariz(v, thres) for v in y_pred_test]
    acc, tn, fp, fn, tp = accConfusion(y_test, y_pred_test)
    #     print('ACC',acc)
    #     print('TN',tn,'FP',fp,'FN',fn, 'TP',tp)
    return acc, tn, fp, fn, tp


def exp(model, optimizer, epochs, eval_loader):
    model.train()
    loss_function = nn.BCELoss()
    losses = []
    for epoch in tqdm(range(epochs)):  # again, normally you would NOT do 300 epochs, it is toy data
        eloss = 0
        for xm, y in train_mask_loader:
            y_label = torch.unsqueeze(y, -1).type(torch.float32)
            model.zero_grad()
            leng = int(xm.shape[1] / 2)
            x = xm[:, :leng]
            masks = xm[:, leng:]

            y_pred = model((x, masks))
            loss = loss_function(y_pred, y_label)
            loss.backward()
            optimizer.step()
        eloss += loss
        #     print('epoch',epoch,'loss',eloss.cpu().detach().numpy())
        losses.append(eloss.cpu().detach().numpy())
    evl(model, eval_loader)
    plt.plot(np.arange(epochs), losses)


def test_crpt(runs, ratio):
    accs = []
    pres = []
    recalls = []
    f1s = []
    for i in range(runs):
        leng = int(X_mask_test.shape[1] / 2)
        X_test = (X_mask_test[:, :leng])
        masks = (X_mask_test[:, leng:])

        X_test_crpt = batch_bit_aug(torch.tensor(X_test), ratio).numpy()
        X_test_mask_crpt = np.concatenate((X_test_crpt, masks), -1)

        dataset_cpt_test = TensorDataset(torch.tensor(X_test_mask_crpt).type(torch.float32).to(cuda1),
                                         torch.tensor(y_test).type(torch.float32).to(cuda1))
        test_cpt_loader = DataLoader(dataset_cpt_test, batch_size=1024, shuffle=False)
        acc, tn, fp, fn, tp = evl(model, test_cpt_loader)
        pre = tp / (tp + fp)
        rec = tp / (tp + fn)
        accs.append(acc)
        pres.append(pre)
        recalls.append(rec)
        f1s.append(2 * pre * rec / (pre + rec))
    return (np.mean(accs), np.std(accs)), (np.mean(pres), np.std(pres)), (np.mean(recalls), np.std(recalls)), (
    np.mean(f1s), np.std(f1s))


path = '../checkpoints/debug/bbheader_gseheader0mask'
# path = '../checkpoints/debug/bbheader_gseheader7800'

# path = '../checkpoints/debug/gseheader7800mask'

# path = '../checkpoints/debug/ipheaders7800'


model = torch.load(path).to(cuda1)
model.replaceMLP()
print(model.mask)
# mlpoptimizer = optim.SGD(model.parameters(), lr=0.1)
mlpoptimizer = optim.SGD(model.usedMLP.parameters(), lr=0.1)
exp(model,mlpoptimizer,50,test_mask_loader)

res = []
for ratio10 in range(1,11):
    ratio =ratio10/50.0
    acc,pre,rec,f1 = test_crpt(10,ratio)
#     print('%5.3f & %5.3f & %5.3f & %5.3f' % (acc[0],pre[0],rec[0],f1[0]))
    res.append([acc[0],pre[0],rec[0],f1[0]])
resnp = np.array(res).T
for a in resnp.tolist():
    output = ''
    for i in a:
        output += '%.3f'%i
        output +=' & '
    print(output)

path = '../log/'+name
torch.save(model.to('cpu'), path)
