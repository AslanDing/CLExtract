import re
import numpy as np

def saveNpy(file):
    with open('./data/'+file+'.txt') as fin:
        data = fin.readline()
        data = re.sub(' |\'','',data)
        start = data.find('[')
        line = data[start+2:-3]
        data = line.split('],[')
    
    array = [d.split(',') for d in data]

    error = 0
    for d in array:
        if len(d)!=10:
            error +=1
    print('error',error,error/len(array))
    array = [a for a in array if len(a)==10]
        
    ress = []
    unique = set()
    for d in array:
        res = ''
        for v in d:
            vb = format(int(v,16),'b')
            vbp = '0'*(8-len(vb))+vb
            res = res+vbp
        if res in unique:
            continue
        unique.add(res)
        nmb = [int(i) for i in res]
        ress.append(nmb)
    nparray = np.array(ress) 
    print('all',len(array),'unique',len(ress))
    with open('./data/binary/'+file+'.npy','wb') as fout:
        np.save(fout,nparray)

names = ['C6_GS2_target2']
name = names[0]
file = 'positive_'+name
saveNpy(file)

file = 'negative_'+name
saveNpy(file)
