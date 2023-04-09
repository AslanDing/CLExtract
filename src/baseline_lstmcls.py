import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
import random
import matplotlib.pyplot as plt
from util.tools import np_bit_loss,npflip,np_bit_aug
from data.loadData import loadData
from util.tools import accConfusion,binariz

from torch.utils.data import DataLoader, TensorDataset,TensorDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.model_set import CNN, MLP, LSTMcls

cuda1 = torch.device('cuda:6')
torch.manual_seed(1)

names = ['bbheader_gseheader']
# names = ['gseheader']
# names = ['ipheaders']

name = names[0]

X,y,_ = loadData(name)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

dataset_train = TensorDataset(torch.tensor(X_train).type(torch.float32).to(cuda1), torch.tensor(y_train).type(torch.float32).to(cuda1))
dataset_test = TensorDataset(torch.tensor(X_test).type(torch.float32).to(cuda1), torch.tensor(y_test).type(torch.float32).to(cuda1))
#dataset_cpt_test = TensorDataset(torch.tensor(X_test_crpt).type(torch.float32).to(cuda1), torch.tensor(y_test).type(torch.float32).to(cuda1))

train_loader = DataLoader(dataset_train, batch_size= 1024,shuffle=True)
test_loader = DataLoader(dataset_test, batch_size= 1024,shuffle=False)
#test_cpt_loader=DataLoader(dataset_cpt_test, batch_size= 1024,shuffle=False)

model = LSTMcls(32,1).to(cuda1)
optimizer = optim.SGD(model.parameters(), lr=0.15)
epochs = 3000

def evl(model,eval_loader):
    model.eval()
    y_test_batch = []
    y_pred_test = []
    with torch.no_grad():
        for x, y in eval_loader:
            y_pred_test.append(model(x.to(cuda1)).cpu())
            y_test_batch.append(y)
    y_test_batch = torch.concat(y_test_batch,0)
    y_pred_test = torch.concat(y_pred_test,0)
    y_pred_test = [binariz(v) for v in y_pred_test]
    acc, tn, fp, fn, tp = accConfusion(y_test, y_pred_test)
    print('ACC',acc)
    print('TN',tn,'FP',fp,'FN',fn, 'TP',tp)
    return acc, tn, fp, fn, tp

def exp(model,optimizer,epochs,eval_loader=test_loader):
    model.train()
    loss_function = nn.BCELoss()
    losses = []
    for epoch in tqdm(range(epochs)):  # again, normally you would NOT do 300 epochs, it is toy data
        eloss = 0
        for x, y in train_loader:
            y_label = torch.unsqueeze(y,-1).type(torch.float32)
            model.zero_grad()
            y_pred = model(x)
            loss = loss_function(y_pred, y_label)
            loss.backward()
            optimizer.step()
        eloss += loss
    #     print('epoch',epoch,'loss',eloss.cpu().detach().numpy())
        losses.append(eloss.cpu().detach().numpy())
    evl(model,eval_loader)
    plt.plot(np.arange(epochs),losses)

exp(model,optimizer,epochs)
model.eval()
for percentage in range(2,22,2):
    rate = percentage/100
    accs= []
    pres= []
    recalls= []
    f1s = []
    for i in range(10):
        X_test_crpt = np.array([np_bit_aug(x,rate) for x in X_test])
        dataset_cpt_test = TensorDataset(torch.tensor(X_test_crpt).type(torch.float32).to(cuda1), torch.tensor(y_test).type(torch.float32).to(cuda1))
        test_cpt_loader=DataLoader(dataset_cpt_test, batch_size= 1024,shuffle=False)

        y_test_batch = []
        y_pred_test = []
        with torch.no_grad():
            for x, y in test_cpt_loader:
                y_pred_test.append(model(x.to(cuda1)).cpu())
                y_test_batch.append(y)
        y_test_batch = torch.concat(y_test_batch,0)
        y_pred_test = torch.concat(y_pred_test,0)
        y_pred_test = [binariz(v) for v in y_pred_test]
        acc, tn, fp, fn, tp = accConfusion(y_test, y_pred_test)

        pre = tp/(tp+fp)
        rec = tp/(tp+fn)
        accs.append(acc)
        pres.append(pre)
        recalls.append(rec)
        f1s.append(2*pre*rec/(pre+rec))

    print('after Corruption')
    print('rate',rate)
    print((np.mean(accs), np.std(accs)),
          (np.mean(pres), np.std(pres)),
          (np.mean(recalls), np.std(recalls)),
          (np.mean(f1s), np.std(f1s)))
"""
bbheader
rate 0.02
(0.7070515036294504, 0.003061597117401371) (0.6607776328925012, 0.002680103725825331) (0.8710401087695446, 0.0025262956051749783) (0.7514756729774585, 0.0023314753224893863)
after Corruption
rate 0.04
(0.6515381956446594, 0.004395346795639629) (0.6068204924669096, 0.0032086776999204042) (0.8938817131203264, 0.0038026828744927082) (0.7228925032015482, 0.0031728653508165286)
after Corruption
rate 0.06
(0.6081921880400967, 0.004400237157043585) (0.5716250834703733, 0.0027407722316860296) (0.915567641060503, 0.005359728985568196) (0.703821314660177, 0.0033472527155649595)
after Corruption
rate 0.08
(0.5782232976149326, 0.003949632037908239) (0.5502246269771895, 0.002368287539820833) (0.9339904826648538, 0.0047142728613421584) (0.6924897002043032, 0.0027508091662614364)
after Corruption
rate 0.1
(0.5592118907708261, 0.003997741388123305) (0.5376199348356238, 0.0023744978841964896) (0.9513256288239292, 0.0025653245770378245) (0.6869947579445298, 0.002144602263734167)
after Corruption
rate 0.12
(0.5446595229865192, 0.002957581832095712) (0.5287038061643252, 0.0016537366442816682) (0.9624065261726716, 0.0035461050593460232) (0.6824791859995913, 0.0018061583786640727)
after Corruption
rate 0.14
(0.5360871068095403, 0.0014538362146431423) (0.5236477250116817, 0.0007896857563875195) (0.9702243371855881, 0.0022505704088710897) (0.6801855617302596, 0.0009765064561408832)
after Corruption
rate 0.16
(0.529519529899758, 0.0025979892739873547) (0.5198918728621986, 0.001365855696841725) (0.9763426240652617, 0.0032574163292211893) (0.6784924458257928, 0.0017950368963998693)
after Corruption
rate 0.18
(0.5248530936743865, 0.0026288494077886617) (0.5172728210239355, 0.0013702258554298742) (0.9813052345343303, 0.003059143439836851) (0.6774448221028792, 0.0017735773389549753)
after Corruption
rate 0.2
(0.5212582094711372, 0.0018093677605672313) (0.5152998152433785, 0.0009583439032290632) (0.9846363018354859, 0.002339177502934755) (0.6765380239454295, 0.0010130843652474183)
"""

"""
gseheader
after Corruption
rate 0.02
(0.5574182163702999, 0.002072210565911948) (0.5390616578494096, 0.0012957080038176717) (0.8685522788203753, 0.0027759319545053673) (0.6652424717895521, 0.0015556294090529627)
after Corruption
rate 0.04
(0.5511334328763404, 0.004012410975831101) (0.5389626813192507, 0.002770314265337829) (0.784798927613941, 0.005730057359753057) (0.6390471947910141, 0.003333623595429803)
after Corruption
rate 0.06
(0.5435726890185965, 0.003625185383758282) (0.5376419403596173, 0.0027376188002658313) (0.7036461126005362, 0.006324123452595408) (0.609534742397623, 0.003605929009684266)
after Corruption
rate 0.08
(0.5403149178770191, 0.005579329388398234) (0.5391848149536205, 0.004519056474705636) (0.6333243967828418, 0.00876560814319668) (0.5824673878058718, 0.006173489514796909)
after Corruption
rate 0.1
(0.5355097054431927, 0.0038500791265398773) (0.5391162720836412, 0.0036256757567545038) (0.5692493297587131, 0.004239062875454628) (0.553770528202784, 0.0037325081030544606)
after Corruption
rate 0.12
(0.5304873082665943, 0.0038805869843468897) (0.5389379353464208, 0.0043926409528370226) (0.5033780160857908, 0.0033295817042917042) (0.520538585361239, 0.00285120279208122)
after Corruption
rate 0.14
(0.5282204425139134, 0.00481187979717216) (0.5407446888720842, 0.005758647840162158) (0.4527077747989276, 0.006492365491885429) (0.4928086181377294, 0.005612568344269911)
after Corruption
rate 0.16
(0.5220442513913397, 0.004993127688538455) (0.5369068259442877, 0.00606967009968384) (0.4066487935656836, 0.010071155394393107) (0.46276148697199027, 0.00864972638478321)
after Corruption
rate 0.18
(0.518813628342609, 0.004472210568684671) (0.5371873804024976, 0.006715160673970063) (0.3585790884718498, 0.004833550422112077) (0.43006751438524543, 0.005256482858458587)
after Corruption
rate 0.2
(0.5170625763540111, 0.003169939478739288) (0.5387979589997408, 0.005354648316621054) (0.3206434316353887, 0.0041272933829331485) (0.4020231744587792, 0.004319098216526638)

"""

"""
ipheader
after Corruption
rate 0.02
(0.6993777452415813, 0.0011922765354979588) (0.6275610216186402, 0.0007582864722220141) (0.9884195193008013, 0.0022199199787513667) (0.7676984535182022, 0.0010166372403029689)
after Corruption
rate 0.04
(0.692203513909224, 0.0030818521024585277) (0.6233763754332052, 0.0022110527422593853) (0.9790968681718863, 0.002932384870968315) (0.7617524663567516, 0.002166249755521456)
after Corruption
rate 0.06
(0.6867496339677892, 0.004355014265808441) (0.6208415318817305, 0.002732237263409225) (0.9676620538965768, 0.006077974084493818) (0.7563870080210545, 0.003561978983810398)
after Corruption
rate 0.08
(0.6808565153733529, 0.002162070194088203) (0.6175689817037864, 0.001396323099165735) (0.958557902403496, 0.0031988625281861214) (0.75117620826199, 0.0017751202850244162)
after Corruption
rate 0.1
(0.6745607613469986, 0.003854477338909297) (0.6141904321784202, 0.0026266439116522564) (0.9478514202476329, 0.0038013076039912232) (0.7453831679763442, 0.002887214061996122)
after Corruption
rate 0.12
(0.6666910688140556, 0.0056945954544932315) (0.6094666983494637, 0.00385619610114767) (0.9375819373634376, 0.0059621002158070785) (0.7387248249691026, 0.004331670038868489)
after Corruption
rate 0.14
(0.6605417276720351, 0.0028409382008685443) (0.6061384095355388, 0.0023605391072866115) (0.9268026219956299, 0.0044690164476744116) (0.7329230300191952, 0.0018376353755480183)
after Corruption
rate 0.16
(0.6551976573938507, 0.008497940531657323) (0.6035278377488172, 0.005908350803315285) (0.9151493080844864, 0.008426333631045993) (0.7273568204448677, 0.006378700295123014)
after Corruption
rate 0.18
(0.6504392386530015, 0.008325925435963609) (0.6006548976936614, 0.005946322381252533) (0.9085943190094683, 0.006911484228686016) (0.7231995764542543, 0.005986298895919184)
after Corruption
rate 0.2
(0.6423865300146414, 0.004836613955829861) (0.5963866699216387, 0.003585029524495005) (0.8924253459577567, 0.007239855736617942) (0.7149583116504136, 0.0037026065587316154)

"""
