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

cuda1 = torch.device('cuda:5')
torch.manual_seed(1)

# names = ['bbheader_gseheader']
names = ['gseheader']
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

model = CNN(128, 10).to(cuda1)
optimizer = optim.SGD(model.parameters(), lr=0.05)
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
(0.9418596612512962, 0.003271211383151324) (0.9606227954939139, 0.002156801829288035) (0.9235214140040787, 0.006927724009784547) (0.9416903188306579, 0.003490115558689027)
after Corruption
rate 0.04
(0.9207051503629451, 0.00292977809891699) (0.9541483802108013, 0.002439254246405128) (0.8866757307953772, 0.00683910135038922) (0.919155992380207, 0.0032912410940405842)
after Corruption
rate 0.06
(0.9007950224680263, 0.005283089712974156) (0.9498767437348331, 0.0034120358535928773) (0.8497620666213461, 0.01167215555749003) (0.8969819058159709, 0.006140206671490672)
after Corruption
rate 0.08
(0.8779467680608365, 0.005644882833541404) (0.9432898933482626, 0.0037266124898476223) (0.8085656016315431, 0.009805296931123185) (0.8707209582287844, 0.006530866317682801)
after Corruption
rate 0.1
(0.855686138956101, 0.005267801816152139) (0.9362516022307629, 0.003801979752004473) (0.7685248130523452, 0.0107969953598019) (0.8440907347232833, 0.006564698128982179)
after Corruption
rate 0.12
(0.8354649153128241, 0.005493741740223661) (0.9269567578070108, 0.004199387007960452) (0.7342624065261726, 0.009193530204164744) (0.8194099722040173, 0.0067269222636905015)
after Corruption
rate 0.14
(0.8199792602834428, 0.003931439255613665) (0.9234290177159139, 0.0054641582645014175) (0.704418762746431, 0.009344848553332194) (0.7991360950841309, 0.0054514530151362865)
after Corruption
rate 0.16
(0.7984445212582094, 0.0033777734133914224) (0.9157308380488949, 0.006984092140949163) (0.6648538409245413, 0.007453144839519098) (0.77033422041601, 0.004520896465076739)
after Corruption
rate 0.18
(0.7810922917386796, 0.005721827658326144) (0.9057290517833616, 0.005308250786227966) (0.635622025832767, 0.00929128460273006) (0.7469822814463567, 0.007642399324675135)
after Corruption
rate 0.2
(0.7668510197027307, 0.004329062220577574) (0.8997774738317788, 0.0056525659737333925) (0.6093813732154997, 0.00911147841074998) (0.726588844393983, 0.006471872135918428)

"""

"""
gseheader 
after Corruption
rate 0.02
(0.7764897515949505, 0.0032753698671059033) (0.9109558032985602, 0.0028618996488166105) (0.6190616621983914, 0.00575059212405619) (0.7371530274270246, 0.004594824904875281)
after Corruption
rate 0.04
(0.7194787566173477, 0.002637839241053618) (0.898569158023745, 0.003124399778167034) (0.5027077747989276, 0.005782997806639194) (0.6446997344556166, 0.004702390238709547)
after Corruption
rate 0.06
(0.6703135604723768, 0.002906748215409074) (0.8719414886877571, 0.005132250121775829) (0.40890080428954423, 0.004380404989919587) (0.5567139180570883, 0.004718796618228164)
after Corruption
rate 0.08
(0.6310981403556399, 0.0022313106528605133) (0.8375015259870529, 0.005913201827504632) (0.3367560321715818, 0.004381471410618908) (0.4803383460119549, 0.0045252121864719184)
after Corruption
rate 0.1
(0.6047780643409799, 0.0020371925977743636) (0.8168801601508721, 0.006257105624620837) (0.28286863270777485, 0.005959285609759681) (0.4201697151293337, 0.006201674922181835)
after Corruption
rate 0.12
(0.5828152572281797, 0.003274807272438988) (0.7812326457958791, 0.010448594717362536) (0.24453083109919574, 0.005273329642650544) (0.3724447429450398, 0.006641619404247016)
after Corruption
rate 0.14
(0.5673815664449572, 0.003963272891797794) (0.7581182489306508, 0.010210312522885184) (0.21369973190348523, 0.0069740659105862884) (0.33337787875488506, 0.00920755140281949)
after Corruption
rate 0.16
(0.5556264422424324, 0.002644710578625458) (0.7358752656486458, 0.007977554489939021) (0.19083109919571045, 0.00557589191264586) (0.3030305330996895, 0.007309786052271728)
after Corruption
rate 0.18
(0.5475091624813357, 0.0024124025049819327) (0.7157156457522449, 0.008966486589518943) (0.17638069705093834, 0.005189520351361022) (0.28297539764330526, 0.006867740133391627)
after Corruption
rate 0.2
(0.5382652368671101, 0.005816664997951618) (0.690192265182352, 0.021334726044012056) (0.15951742627345844, 0.00858578826863189) (0.25910729565045443, 0.012727375302917282)
"""

"""
ipheader
rate 0.02
(0.9606881405563691, 0.002531715441550587) (0.9848299880224097, 0.0018055102325058318) (0.9361981063364894, 0.004135490041246433) (0.9598945709618079, 0.002637918957377249)
after Corruption
rate 0.04
(0.9510248901903369, 0.0028718953507954753) (0.9839124201954178, 0.0008416887466093037) (0.9175528040786599, 0.0058157217123604235) (0.9495647694200375, 0.003105296508813053)
after Corruption
rate 0.06
(0.9350658857979501, 0.003843163908757443) (0.981469728702602, 0.0022594764989616366) (0.8875455207574655, 0.006147399226805301) (0.9321404472273436, 0.004171796727356861)
after Corruption
rate 0.08
(0.9186676427525621, 0.0037104907459660726) (0.9788640546991312, 0.002301945518978394) (0.8566642388929353, 0.006931028261250008) (0.9136815781833068, 0.004227417021637694)
after Corruption
rate 0.1
(0.8982064421669108, 0.004181267189714947) (0.9763421901541071, 0.0023957575986838266) (0.8172614712308812, 0.008509026567528805) (0.8897205090957125, 0.0050444868339681505)
after Corruption
rate 0.12
(0.8754758418740849, 0.006762785610472028) (0.9734232951547972, 0.0038257612826027663) (0.7733430444282593, 0.013209944994907073) (0.8618614747030863, 0.008457575856921501)
after Corruption
rate 0.14
(0.8478038067349927, 0.00467261300065385) (0.9685655207362063, 0.0026354496967849355) (0.7205389657683904, 0.008475295159261199) (0.8263186256063595, 0.006086918988140587)
after Corruption
rate 0.16
(0.8244143484626647, 0.004566022595818113) (0.9655141709451774, 0.0050867312462868535) (0.6747268754552075, 0.00766100148131375) (0.7943206995297063, 0.006002508624361122)
after Corruption
rate 0.18
(0.804575402635432, 0.0065117737272048035) (0.9662066443623551, 0.00614793812325964) (0.6333576110706483, 0.014424653941065835) (0.765028014375794, 0.01006459211388803)
after Corruption
rate 0.2
(0.7784407027818447, 0.006637305589780673) (0.9649675088323967, 0.0065072577197790755) (0.580189366351056, 0.010935658778804635) (0.7246249289155683, 0.009656670672893221)

"""
