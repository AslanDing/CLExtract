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
from models.informer import Informer, InformerStack

cuda1 = torch.device('cuda:5')
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

model = Informer(1,16,1).to(cuda1)
optimizer = optim.SGD(model.parameters(), lr=0.2)
epochs = 1000

# from torchscan import summary
# # from torchsummary import summary
#
# y_test_batch = []
# y_pred_test = []
# with torch.no_grad():
#     for x, y in test_loader:
#         y_pred_test.append(model(x.to(cuda1)).cpu())
#         y_test_batch.append(y)
#
# model.mask = False
# model.eval()
# summary(model,(480,))
# print("")


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
        test_cpt_loader=DataLoader(dataset_cpt_test, batch_size= 512,shuffle=False)

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
(0.9502938126512271, 0.004895237045862565) (0.9772080487409772, 0.0022994166879247717) (0.9237933378653977, 0.009451310265139685) (0.9497247588429589, 0.0051860570406911215)
after Corruption
rate 0.04
(0.9142412720359487, 0.006045527570409901) (0.9726355591633867, 0.0035665676684602118) (0.8554044867437117, 0.010761839865443542) (0.910228870970276, 0.006729746578910089)
after Corruption
rate 0.06
(0.8805392326304873, 0.005183085281085819) (0.9680009867197146, 0.003813776821610811) (0.7912304554724677, 0.01032975186155274) (0.8706919411549023, 0.006328027822723734)
after Corruption
rate 0.08
(0.8505357760110611, 0.007021618486329546) (0.9638436830125225, 0.002527931899700138) (0.7335825968728755, 0.01466388974763033) (0.8330080723794977, 0.009258054238971837)
after Corruption
rate 0.1
(0.8220532319391634, 0.006063978502478628) (0.9598722003211536, 0.006300702673436426) (0.6783820530251529, 0.008471475755069539) (0.7949285805418346, 0.007537439403629001)
after Corruption
rate 0.12
(0.7887659868648462, 0.004753197360400623) (0.9531421337921546, 0.005621141436428708) (0.6148198504418763, 0.009272364438682549) (0.7474320938799849, 0.006924029414138167)
after Corruption
rate 0.14
(0.7722779122018666, 0.007315957675852567) (0.9529480818035972, 0.005711765559541287) (0.5808293677770224, 0.014048229887792795) (0.7216496893439366, 0.011175246915496696)
after Corruption
rate 0.16
(0.7452817144832353, 0.005123381677304785) (0.9535420194198927, 0.0037884635198318036) (0.5246091094493541, 0.009940342168552592) (0.6767875614488237, 0.00847775333075147)
after Corruption
rate 0.18
(0.7191496716211545, 0.0038592037468223486) (0.9421734937667076, 0.0076330114941888265) (0.47702243371855885, 0.010201902357191338) (0.6332641260434083, 0.007995930033360642)
after Corruption
rate 0.2
(0.7020739716557207, 0.006994766422479789) (0.9382036673259286, 0.004395189052860473) (0.44323589394969415, 0.013203016599657605) (0.601948043742723, 0.012939336937460915)
"""

"""
gseheader
after Corruption
rate 0.02
(0.7501561015338674, 0.002432141865554495) (0.8078252806760172, 0.004194910866570674) (0.6646916890080429, 0.0038123475224341036) (0.7292895872754792, 0.0026552885834296713)
after Corruption
rate 0.04
(0.693864531016696, 0.004004251666364562) (0.7759548305116628, 0.004569785048077953) (0.5558445040214478, 0.005933418644110088) (0.6477020692255244, 0.005329212257726551)
after Corruption
rate 0.06
(0.6501153793945975, 0.005752001170811479) (0.741557048638273, 0.008632364973108918) (0.4742359249329759, 0.007268778343873673) (0.5784927025491422, 0.007544109448105304)
after Corruption
rate 0.08
(0.6229944346409664, 0.003087730084563684) (0.7218636077934075, 0.005288289238824583) (0.4154959785522788, 0.005555228910639955) (0.527395257302024, 0.0050450269996768615)
after Corruption
rate 0.1
(0.5993620198181078, 0.003963179909607203) (0.6972639226692314, 0.007380265476001477) (0.3688739946380697, 0.004665880759052488) (0.48248693964315326, 0.005478282167236915)
after Corruption
rate 0.12
(0.5818514999321298, 0.004499647952712741) (0.6790641420343564, 0.009052220273224689) (0.33018766756032175, 0.005977049306746907) (0.4443086578606271, 0.0068613633387606164)
after Corruption
rate 0.14
(0.5709651147006923, 0.004170149429787023) (0.6682635104724984, 0.007740366192362954) (0.303029490616622, 0.0075630307165288685) (0.41694354054776034, 0.008254683093002974)
after Corruption
rate 0.16
(0.5575539568345323, 0.005013309017508745) (0.6486450973547632, 0.010204404475055216) (0.27512064343163534, 0.008767862782784967) (0.3863207423499654, 0.009982124010067926)
after Corruption
rate 0.18
(0.5495995656305145, 0.0028078596673535867) (0.6382104949458214, 0.006177552314443208) (0.25498659517426275, 0.0072885280802944844) (0.36433003888066245, 0.007763440148837052)
after Corruption
rate 0.2
(0.5439120401791774, 0.0038239319911299715) (0.6295180725651608, 0.008765873949551863) (0.24101876675603218, 0.007419059416475217) (0.3485353040495053, 0.008542814738366591)

"""

"""
ipheader
after Corruption
rate 0.02
(0.948316251830161, 0.00205630628187235) (0.9400093192858507, 0.003541511721556554) (0.958339402767662, 0.0034899194612413413) (0.9490770621206579, 0.002012762131931778)
after Corruption
rate 0.04
(0.9445827232796485, 0.0029564890931637806) (0.9378657991189974, 0.003920096536980766) (0.9528769118718134, 0.004321778208054422) (0.9453028060444664, 0.002921074651265144)
after Corruption
rate 0.06
(0.9372254758418741, 0.004531265886947265) (0.9317386137124236, 0.0041936400338170735) (0.9442825928623453, 0.007175088011787532) (0.9379547202007155, 0.004595773949006297)
after Corruption
rate 0.08
(0.9341508052708638, 0.0024849786343328927) (0.9299576142526467, 0.003105953063017625) (0.9397669337217771, 0.004272398944227959) (0.9348284442041898, 0.0025118356068403506)
after Corruption
rate 0.1
(0.929575402635432, 0.0034109803095817835) (0.9268056777211925, 0.005460439701653127) (0.9336489439184268, 0.0057850847610518685) (0.9301929586757283, 0.0033833906285641773)
after Corruption
rate 0.12
(0.9201683748169838, 0.002192835402745274) (0.9186808941408303, 0.003953514888763002) (0.922869628550619, 0.004668716827541825) (0.9207555469662971, 0.0022137427713471735)
after Corruption
rate 0.14
(0.9148609077598829, 0.004007007960001183) (0.9136724880709574, 0.0029467406990269568) (0.9172614712308812, 0.0072993269141252955) (0.9154490916833342, 0.004219807105052416)
after Corruption
rate 0.16
(0.9075402635431917, 0.003005924621910381) (0.90929554709491, 0.003909790115337998) (0.9064821558630737, 0.007428987618353978) (0.9078597651427287, 0.0032699490202384466)
after Corruption
rate 0.18
(0.9, 0.0052759451674164876) (0.9027780215836246, 0.006193131572461444) (0.8977421704297159, 0.008407110869924588) (0.9002245399060186, 0.005397599028088272)
after Corruption
rate 0.2
(0.8904465592972182, 0.0054144318041329615) (0.9002440942036891, 0.004703021984974747) (0.8794610342316096, 0.0075215940909214045) (0.8897220046279152, 0.005633971665885769)

"""
