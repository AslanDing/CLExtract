import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset,TensorDataset

import numpy as np
import random
import matplotlib.pyplot as plt
from util.tools import npcutout,npflip, torchflip,batch_flip,batch_bit_aug
from data.loadData import loadData
from util.tools import accConfusion,binariz,CustomTensorDataset
cuda1 = torch.device('cuda:0')

torch.manual_seed(1)

name = 'bbheader_gseheader'
batch_size = 1024
X,y,mask = loadData(name)
X_mask = np.concatenate([X,mask],-1)
print(X_mask.shape)
flog = open('../logs/'+name+'/loss.log','w')


class SimCLR(object):

    def __init__(self, model,
                 lr=0.0003, weight_decay=1e-4, n_views=2, device='cpu',
                 batch_size=32, temperature=1.0, fp16_precision=False):
        self.model = model
        self.n_views = n_views
        self.device = device
        self.batch_size = batch_size
        self.temperature = temperature
        self.fp16_precision = fp16_precision
        self.optimizer = optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(data_loader), eta_min=0,
                                                                    last_epoch=-1)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader, epochs):

        scaler = GradScaler(enabled=self.fp16_precision)

        n_iter = 0
        losses = []
        for epoch_counter in tqdm(range(epochs)):
            eloss = 0

            for xm in data_loader:
                leng = int(xm.shape[1] / 2)
                xs = xm[:, :leng]
                masks = xm[:, leng:]
                augxs = [batch_bit_aug(xs, 0.2, 0.5, cuda1) for _ in range(2)]
                augmasks = [masks for _ in range(2)]

                xs = torch.cat(augxs, dim=0)
                masks = torch.cat(augmasks, dim=0)

                features = self.model((xs, masks))
                logits, labels = self.info_nce_loss(features)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                n_iter += 1
                eloss += loss
            flog.write(str(eloss.cpu().detach().numpy()) + '\n')
            flog.flush()
            # warmup for the first 10 epochs
            losses.append(eloss.cpu().detach().numpy())
            if epoch_counter >= 10:
                self.scheduler.step()
            if epoch_counter % 200 == 0:
                path = '../checkpoints/debug/' + name + str(epoch_counter)
                if mask:
                    path += 'mask'
                torch.save(model, path)
        # plt.plot(losses)

data_loader = DataLoader(torch.tensor(X_mask).type(torch.float32).to(cuda1), batch_size= batch_size,shuffle=True,drop_last=True)
print(X_mask.shape)
mask = True
from models.model_set import paddingCNN
model = paddingCNN(128, 10,output_dims=128,target=128,mask=mask).to(cuda1)
simclr = SimCLR(model=model,lr = 1e-3, device = cuda1,batch_size=batch_size,temperature=0.07)
simclr.train(data_loader,epochs=8000)
path = '../checkpoints/'+name
if mask:
    path+='mask'
torch.save(model.to('cpu'), path)
flog.close()