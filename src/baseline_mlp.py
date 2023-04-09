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

cuda1 = torch.device('cuda:7')
torch.manual_seed(1)

# names = ['bbheader_gseheader']
# names = ['gseheader']
names = ['ipheaders']

name = names[0]

X,y,_ = loadData(name)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

dataset_train = TensorDataset(torch.tensor(X_train).type(torch.float32).to(cuda1), torch.tensor(y_train).type(torch.float32).to(cuda1))
dataset_test = TensorDataset(torch.tensor(X_test).type(torch.float32).to(cuda1), torch.tensor(y_test).type(torch.float32).to(cuda1))
#dataset_cpt_test = TensorDataset(torch.tensor(X_test_crpt).type(torch.float32).to(cuda1), torch.tensor(y_test).type(torch.float32).to(cuda1))

train_loader = DataLoader(dataset_train, batch_size= 1024,shuffle=True)
test_loader = DataLoader(dataset_test, batch_size= 1024,shuffle=False)
#test_cpt_loader=DataLoader(dataset_cpt_test, batch_size= 1024,shuffle=False)

model = MLP(480, 16).to(cuda1)
optimizer = optim.SGD(model.parameters(), lr=0.1)
epochs = 1000

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
after Corruption
rate 0.02
(0.9608019357068786, 0.0035527591171875816) (0.9758950837301766, 0.00187512320278883) (0.9462950373895309, 0.007956967988252624) (0.9608452705967465, 0.003710598400790777)
after Corruption
rate 0.04
(0.9371240926374005, 0.002868994123746961) (0.9742559727638239, 0.002273164787493706) (0.9001359619306595, 0.005827757218259087) (0.9357179713555285, 0.0031126481043001765)
after Corruption
rate 0.06
(0.9154856550293813, 0.004601979590630605) (0.9723547315171348, 0.002945314587546644) (0.8581917063222297, 0.008664306711817707) (0.9116893113414152, 0.0051652968566546055)
after Corruption
rate 0.08
(0.8921880400967854, 0.004847777810924958) (0.9693140945332417, 0.0023652013545566886) (0.8137321549966009, 0.009536738499932126) (0.8847043246465164, 0.005756277681005419)
after Corruption
rate 0.1
(0.8669892844797789, 0.003462835751501744) (0.9696537293626593, 0.003466620413622594) (0.7622705642420122, 0.005792760133996653) (0.8535336703798089, 0.004141259452002251)
after Corruption
rate 0.12
(0.8430694780504666, 0.005381680043146492) (0.965330717881202, 0.003342879691810969) (0.7171312032630863, 0.010690317517388935) (0.8228761403386379, 0.007119479640505684)
after Corruption
rate 0.14
(0.8213619080539234, 0.003245542922725318) (0.9626772065657698, 0.0034882669432323254) (0.6748470428280082, 0.0060731776160404885) (0.7934480006005428, 0.004363356704118875)
after Corruption
rate 0.16
(0.7984445212582095, 0.006579300758915608) (0.9594592467191662, 0.003669274842884153) (0.6302515295717199, 0.013645220916217721) (0.760675831536831, 0.009782136746096851)
after Corruption
rate 0.18
(0.7827860352575182, 0.007201391864708488) (0.9579335041290207, 0.0055664495617842955) (0.5991162474507139, 0.013314727642601762) (0.7371008870456028, 0.010651095149129796)
after Corruption
rate 0.2
(0.7594884203249224, 0.006730740139750105) (0.9554956480320855, 0.005122761219119824) (0.5527532290958532, 0.013587522728310404) (0.7002479415080904, 0.011030924451856139)

"""

"""
gseheader
after Corruption
rate 0.02
(0.8131125288448487, 0.0027302398450983222) (0.784500065290112, 0.002893947849068252) (0.8698391420911529, 0.004045060257698287) (0.8249623573571137, 0.0026042718010400117)
after Corruption
rate 0.04
(0.7870639337586536, 0.002880801551252572) (0.791819288925759, 0.0034223411051295376) (0.7861394101876675, 0.0043329738475676996) (0.7889608281184172, 0.002950729973884611)
after Corruption
rate 0.06
(0.7598208225872133, 0.0035615773283551957) (0.7943744538202009, 0.003986625205704876) (0.7092225201072386, 0.005147453083109924) (0.7493785816757994, 0.0039671710686967196)
after Corruption
rate 0.08
(0.7333378580154745, 0.003604286362492387) (0.792540259726968, 0.005476236168505376) (0.6412064343163539, 0.005598083772927468) (0.7088658770620676, 0.0041577260222753895)
after Corruption
rate 0.1
(0.7117551242025248, 0.004930668431474105) (0.789098165826651, 0.0062940010505054805) (0.5878016085790885, 0.006206014496597398) (0.673726393447078, 0.005924687748193499)
after Corruption
rate 0.12
(0.6859915840912175, 0.002745885661156028) (0.7749095970861181, 0.0035027050809698233) (0.5353083109919571, 0.00474470240447931) (0.6331905094137525, 0.003926613472154949)
after Corruption
rate 0.14
(0.6680331206732728, 0.0035553638251438974) (0.7665499456700018, 0.006892840302055413) (0.495201072386059, 0.004817463864116433) (0.6016752783231581, 0.0043331070743256195)
after Corruption
rate 0.16
(0.6461517578390119, 0.0027017467744536) (0.7500151818061955, 0.004139315346179831) (0.4516890080428954, 0.005577180808415063) (0.5638004474855152, 0.004667729884024043)
after Corruption
rate 0.18
(0.627460295914212, 0.003692664588058243) (0.7333813169554467, 0.008350134743016161) (0.415254691689008, 0.005834965283344007) (0.5302208507308157, 0.005087577950137358)
after Corruption
rate 0.2
(0.6129903624270395, 0.00468745059164013) (0.720775303692887, 0.007770648915827927) (0.3846380697050938, 0.007942220675027808) (0.5015641966379374, 0.007880995457448402)

"""

"""
after Corruption
rate 0.02
(0.9773060029282575, 0.0018735701159692173) (0.9837678506452476, 0.0017139347746749928) (0.9708667152221413, 0.0036561982222902112) (0.9772699879362877, 0.0019071855902167035)
after Corruption
rate 0.04
(0.9663616398243045, 0.002034362112970745) (0.9836181705130222, 0.0015318162227162182) (0.9488710852148581, 0.003653295325268591) (0.9659284089795479, 0.002108462217010995)
after Corruption
rate 0.06
(0.9551244509516839, 0.002091194268973016) (0.9838367550735126, 0.0023707305818355968) (0.9259286234522943, 0.004641367506585525) (0.9539949538805228, 0.0022404060367539973)
after Corruption
rate 0.08
(0.941398243045388, 0.003209624566462361) (0.9852049317209854, 0.0020861160756503357) (0.8968681718863802, 0.006409322651128905) (0.9389496379462813, 0.00353295421967856)
after Corruption
rate 0.1
(0.9213396778916545, 0.004651490328078517) (0.9824973586283322, 0.002733305089807965) (0.8587764020393298, 0.008027858866699765) (0.9164640414767057, 0.005220233217668608)
after Corruption
rate 0.12
(0.9059297218155198, 0.0035374698179647386) (0.983713801342447, 0.0027929073995932433) (0.8265112891478514, 0.007014711800631894) (0.898267620762368, 0.004182656315582049)
after Corruption
rate 0.14
(0.8826500732064421, 0.004845470203037448) (0.9852443633758492, 0.0024415723066602978) (0.778150036416606, 0.009115506393102416) (0.8695088561367923, 0.0059924458649431795)
after Corruption
rate 0.16
(0.8585285505124451, 0.004930577519561156) (0.9811829108595239, 0.0024353989159439276) (0.732556445739257, 0.010054149411063319) (0.8387927052307361, 0.0065565850507890745)
after Corruption
rate 0.18
(0.8337847730600293, 0.00612806138196881) (0.9802890906812683, 0.0038947537600814756) (0.6830298616168973, 0.01323401707645609) (0.8050094766685241, 0.008769582521835429)
after Corruption
rate 0.2
(0.8104319180087849, 0.005459772578934237) (0.9806762059343139, 0.003940940120003771) (0.6353241077931536, 0.010649849475557806) (0.7710443962920537, 0.007976276469887673)

"""
