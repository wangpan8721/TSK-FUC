import argparse
import collections
import os
import copy
import numpy
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import scipy.io as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random
from lib.inits import *
from lib.models import *
from lib.tuning_train import *
import torch.nn.functional as F
import torch.utils.data as Data
from ucimlrepo import fetch_ucirepo
from PIL import Image
from keras.datasets import mnist
import argparse
import trainModel
import time
from lib.optim import AdaBound
import torch
torch.set_printoptions(threshold=np.inf)

np.random.seed(1447)
t.manual_seed(1447)


def get_parser():
    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument('--gpu', dest='gpu', action='store_true')
    flag_parser.add_argument('--cpu', dest='gpu', action='store_false')
    parser.set_defaults(gpu=True)

    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument('--bn', dest='bn', action='store_true')
    flag_parser.add_argument('--no_bn', dest='bn', action='store_false')
    parser.set_defaults(bn=True)

    # training parameters
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--epochs', default=200, type=int, help='total training epochs')
    parser.add_argument('--patience', default=40, type=int, help='training patience for early stopping')
    parser.add_argument('--data', default='Abalone', type=str, help='dataset name')
    parser.add_argument('--n_rules', default=20, type=int, help='number of rules of TSK')
    parser.add_argument('--loss_type', default='crossentropy', type=str, help='loss type')
    parser.add_argument('--optim_type', default='adabound', type=str, help='optimization type')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='l2 loss weight')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')

    # weights of regularization
    parser.add_argument('--weight_frs', default=0, type=float, help='UR loss weight')

    parser.add_argument('--init', default='kmean', type=str, help='Initilization approach for TSK rule centers')
    parser.add_argument('--tune_param', default=1, type=int, help='whether to tune parameter')
    parser.add_argument('--repeats', default=1, type=int, help='repeat to get best stop pos for without earlystopping')

    return parser.parse_args()


parser = argparse.ArgumentParser()
args = get_parser()
# args.data='Yeast'
# args.batch_size=512
# args.epochs=2000
# args.patience=150

# args.lr=0.01
data_root = 'data/'

class Noise(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self):
        return self.noise


def loaddata(src):
    # if src=='Yeast':
    #     f = np.load(os.path.join('data/', src + '.npz'))
    #     data = f['con_data'].astype('float')
    #     label = f['label']
    #     n_classes = len(np.unique(label))
    #     train_idx, test_idx = f['trains'], f['tests']
    #
    #     test_index = []
    #     train_index = []
    #     for i in range(len(test_idx)):
    #         for j in range(len(test_idx[i])):
    #             test_index.append(test_idx[i][j])
    #     for i in range(len(train_idx)):
    #         for j in range(len(train_idx[i])):
    #             train_index.append(train_idx[i][j])
    #     # print(test_index)
    #     x_train, y_train = data[train_index], label[train_index]
    #     x_test, y_test = data[test_index], label[test_index]
    if src == 'Yeast':
        plates_faults = fetch_ucirepo(id=110)
        x = plates_faults.data.features
        y = plates_faults.data.targets

        x = x.values
        y = y.values
        y2 = []
        for i in range(len(y)):
            if y[i][0] == 'CYT':
                y2.append(0)
            elif y[i][0] == 'ERL':
                y2.append(1)
            elif y[i][0] == 'EXC':
                y2.append(2)
            elif y[i][0] == 'ME1':
                y2.append(3)
            elif y[i][0] == 'ME2':
                y2.append(4)
            elif y[i][0] == 'ME3':
                y2.append(5)
            elif y[i][0] == 'MIT':
                y2.append(6)
            elif y[i][0] == 'NUC':
                y2.append(7)
            elif y[i][0] == 'POX':
                y2.append(8)
            elif y[i][0] == 'VAC':
                y2.append(9)
        print(numpy.unique(y, return_counts=True))

        print(y)
        print(numpy.unique(y, return_counts=True))
        print(numpy.unique(y2, return_counts=True))
        print(x[0])
        print(x)
        print(y2)
        print(len(np.unique(y2)))
        print(len(x))
        print(len(y2))
        return x, np.asarray(y2)

    elif src == 'Steel':
        # fetch dataset
        # Steel data (as pandas dataframes)
        plates_faults = fetch_ucirepo(id=198)
        x = plates_faults.data.features
        y = plates_faults.data.targets
        x = x.values
        y = y.values
        y2 = []
        for i in y:
            for j in range(len(i)):
                if i[j] == 1:
                    y2.append(j)
                    break
        print(numpy.unique(y2, return_counts=True))
        return x, np.asarray(y2)

    elif src == 'Vehicle':
        plates_faults = fetch_ucirepo(id=149)
        x = plates_faults.data.features
        y = plates_faults.data.targets

        x = x.values
        y = y.values
        y2 = []
        for i in range(len(y)):
            if y[i][0] == 'bus':
                y2.append(0)
            elif y[i][0] == 'opel':
                y2.append(1)
            elif y[i][0] == 'saab':
                y2.append(2)
            elif y[i][0] == 'van':
                y2.append(3)
            elif y[i][0] == '204':
                x[i] = x[i - 1]
                y2.append(3)
        print(numpy.unique(y, return_counts=True))

        # print(len(x))
        # print(len(y))
        # print(len(y2))
        # print(numpy.unique(y2, return_counts=True))
        return x, np.asarray(y2)
    elif src == 'IS':
        plates_faults = fetch_ucirepo(id=50)
        x = plates_faults.data.features
        y = plates_faults.data.targets

        x = x.values
        y = y.values
        y2 = []
        for i in range(len(y)):
            if y[i][0] == 'BRICKFACE':
                y2.append(0)
            elif y[i][0] == 'CEMENT':
                y2.append(1)
            elif y[i][0] == 'FOLIAGE':
                y2.append(2)
            elif y[i][0] == 'GRASS':
                y2.append(3)
            elif y[i][0] == 'PATH':
                y2.append(4)
            elif y[i][0] == 'SKY':
                y2.append(5)
            elif y[i][0] == 'WINDOW':
                y2.append(6)
        print(numpy.unique(y, return_counts=True))
        # print(x)
        # print(y)
        # print(y2)
        print(len(x))
        print(len(y))
        print(len(y2))
        return x, np.asarray(y2)

    elif src == 'Abalone':
        plates_faults = fetch_ucirepo(id=1)
        x = plates_faults.data.features
        y = plates_faults.data.targets
        x = x.values
        y = y.values
        y2 = []
        for i in range(len(x)):
            if y[i][0] < 11:
                y2.append(0)
            elif y[i][0] < 21:
                y2.append(1)
            elif y[i][0] < 31:
                y2.append(2)
            # elif y[i][0]<17:
            #     y2.append(3)
            # elif y[i][0]<21:
            #     y2.append(4)
            # elif y[i][0]<25:
            #     y2.append(5)
            # elif y[i][0]<30:
            #     y2.append(6)

            if x[i][0] == 'M':
                x[i][0] = 0.0
            else:
                x[i][0] = 1.0
        return x, np.asarray(y2)

    elif src == 'Waveform21':
        # Waveform21 data
        plates_faults = fetch_ucirepo(id=107)
        x = plates_faults.data.features
        y = plates_faults.data.targets
        # print(plates_faults.metadata)
        # print(plates_faults.variables)
        x = x.values
        y = y.values
        y2 = []
        for i in range(len(x)):
            y2.append(y[i][0])
        return x, np.asarray(y2)

    elif src == 'Page_blocks':
        plates_faults = fetch_ucirepo(id=78)
        x = plates_faults.data.features
        y = plates_faults.data.targets

        # print(plates_faults.metadata)
        # print(plates_faults.variables)

        x = x.values
        y = y.values
        y2 = []
        for i in range(len(x)):
            y2.append(y[i][0] - 1)
        return x, np.asarray(y2)
    elif src == 'Satellite':
        plates_faults = fetch_ucirepo(id=146)
        x = plates_faults.data.features
        y = plates_faults.data.targets

        # print(plates_faults.metadata)
        # print(plates_faults.variables)

        x = x.values
        y = y.values
        y2 = []
        for i in range(len(x)):
            if y[i][0] == 7:
                y2.append(5)
            else:
                y2.append(y[i][0] - 1)
        return x, np.asarray(y2)
    elif src == 'Magic':
        plates_faults = fetch_ucirepo(id=159)
        x = plates_faults.data.features
        y = plates_faults.data.targets

        # print(plates_faults.metadata)
        # print(plates_faults.variables)

        x = x.values
        y = y.values
        y2 = []
        for i in range(len(x)):
            if y[i][0] == 'h':
                y2.append(0)
            else:
                y2.append(1)

        # print(y)
        # print(numpy.unique(y, return_counts=True))
        # print(numpy.unique(y2, return_counts=True))
        # print(x[0])
        # print(x)
        # print(y2)
        # print(len(np.unique(y2)))
        # print(len(x))
        # print(len(y2))
        return x, np.asarray(y2)
    #图片
    elif src=='cifar10':
        x=[]
        y=[]
        path1 = './data/cifar10/test'
        path2 = './data/cifar10/train'
        paths = [path1, path2]
        for i in range(2):
            filelist = os.listdir(paths[i])
            for j in range(len(filelist)):
                # print(item)
                for root, dirs, files in os.walk(paths[i] + '/' + filelist[j]):
                    for file in files:
                        img = np.array(Image.open(paths[i]+'/'+filelist[j]+'/'+file))
                        x.append(img)
                        y.append(j)
        # 降为2维
        x=fast_list2arr(x)
        x = (x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
        # print(x)
        # print(y)
        return x,np.array(y)
    elif src=='Minist':

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x=np.concatenate((x_train,x_test),axis=0)
        y=np.concatenate((y_train,y_test),axis=0)
        x = (x.reshape(x.shape[0], x.shape[1] * x.shape[2]))
        # print(x)
        # print(x[0])
        # print(len(y))
        return x,y
    # 多标签分类
    elif src == 'AI4I2020':
        plates_faults = fetch_ucirepo(id=601)
        x = np.array(plates_faults.data.features)
        y = fast_list2arr(np.array(plates_faults.data.targets))
        # le=LabelEncoder()
        # le=le.fit(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
        # y=le.transform(y)

        print(len(y))
        print(type(y))
        print(y)
        print(numpy.unique(y, return_counts=True))
        print(x)
        print(x[0])
        print(type(x))
        return x, y
        plates_faults = fetch_ucirepo(id=59)
        x = np.array(plates_faults.data.features)
        y = fast_list2arr(np.array(plates_faults.data.targets))
        le = LabelEncoder()
        le = le.fit(
            ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y', 'Z'])
        y = le.transform(y)

        print(len(y))
        print(type(y))
        print(y)
        print(numpy.unique(y, return_counts=True))
        print(x)
        print(x[0])
        print(type(x))
        return x, y
    #>5000
    elif src=='HandwrittenDigits':
        plates_faults = fetch_ucirepo(id=80)
        x = np.array(plates_faults.data.features)
        y=np.array(list(np.array(plates_faults.data.targets).flatten()))

        # print(len(y))
        # print(type(y))
        # print(y)
        # print(numpy.unique(y, return_counts=True))
        # print(x)
        # print(type(x))
        # print(len(x[0]))
        return x, y
    elif src=='PenBasedHandwrittenDigits':
        plates_faults = fetch_ucirepo(id=81)
        x = np.array(plates_faults.data.features)
        y=np.array(list(np.array(plates_faults.data.targets).flatten()))
        #
        # print(len(y))
        # print(type(y))
        # print(y)
        # print(numpy.unique(y, return_counts=True))
        # print(len(x[0]))
        # print(type(x))
        return x, y
    elif src=='LetterRecognition':
        plates_faults = fetch_ucirepo(id=59)
        x = np.array(plates_faults.data.features)
        y = fast_list2arr(np.array(plates_faults.data.targets))
        le=LabelEncoder()
        le=le.fit(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
        y=le.transform(y)
        #
        # print(len(y))
        # print(type(y))
        # print(y)
        # print(numpy.unique(y, return_counts=True))
        # print(len(x[0]))
        # print(type(x))
        return x, y
    elif src=='RoomOccupancy':
        plates_faults = fetch_ucirepo(id=864)
        x = np.array(plates_faults.data.features)
        x=np.delete(x,[0,1],1)
        y=np.array(list(np.array(plates_faults.data.targets).flatten()))
        #
        # print(len(y))
        # print(type(y))
        # print(y)
        # print(numpy.unique(y, return_counts=True))
        # print(len(x[0]))
        # print(type(x))
        return x, y
    elif src=='DryBean':
        plates_faults = fetch_ucirepo(id=602)
        x = np.array(plates_faults.data.features)
        y = fast_list2arr(np.array(plates_faults.data.targets))
        le = LabelEncoder()
        le = le.fit(['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA'])
        y = le.transform(y)
        #
        # print(len(y))
        # print(type(y))
        # print(y)
        # print(numpy.unique(y, return_counts=True))
        # print(len(x[0]))
        # print(type(x))
        return x, y

    elif src=='Nursery':
        plates_faults = fetch_ucirepo(id=76)

        x = plates_faults.data.features
        for i in x.columns:
            class_encoder = LabelEncoder()
            x[i]=class_encoder.fit_transform(x[i].values)

        x=np.array(x)
        y=np.array(plates_faults.data.targets)
        le = LabelEncoder()
        le = le.fit(['not_recom', 'priority', 'recommend', 'spec_prior', 'very_recom'])
        y = le.transform(y)
        #
        # print(len(y))
        # print(type(y))
        # print(y)
        # print(numpy.unique(y, return_counts=True))
        # print(len(x[0]))
        # print(type(x))
        return x, y
    elif src=='Shuttle':
        plates_faults = fetch_ucirepo(id=148)
        x=np.array(plates_faults.data.features)
        y=np.array(list(np.array(plates_faults.data.targets).flatten()))
        y=y-1
        # print(len(y))
        # print(type(y))
        # print(y)
        # print(numpy.unique(y, return_counts=True))
        # print(len(x[0]))
        # print(type(x))
        return x, y
    elif src=='Connect-4':
        plates_faults = fetch_ucirepo(id=26)
        x = plates_faults.data.features
        for i in x.columns:
            class_encoder = LabelEncoder()
            x[i] = class_encoder.fit_transform(x[i].values)

        x = np.array(x)

        y=np.array(list(np.array(plates_faults.data.targets).flatten()))
        le = LabelEncoder()
        le = le.fit(['draw', 'loss', 'win'])
        y = le.transform(y)

        return x, y



    #太大
    elif src=='Covertype':
        plates_faults = fetch_ucirepo(id=31)
        x = np.array(plates_faults.data.features)
        y=np.array(list(np.array(plates_faults.data.targets).flatten()))
        y=y-1
        # print(len(y))
        # print(type(y))
        # print(y)
        # print(numpy.unique(y, return_counts=True))
        # print(len(x[0]))
        # print(type(x))
        return x, y
    elif src=='PokerHand':
        plates_faults = fetch_ucirepo(id=158)
        x=np.array(plates_faults.data.features)
        y=np.array(list(np.array(plates_faults.data.targets).flatten()))
        # y=y-1
        # print(len(y))
        # print(type(y))
        # print(y)
        # print(numpy.unique(y, return_counts=True))
        # print(len(x[0]))
        # print(type(x))
        return x, y

def prepare(src,fnums,rule,acc,lrf1,lrf2,sample,Theta1,Theta2):
    print(src)

    x, y = loaddata(src)
    n_classes = len(np.unique(y))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    print(numpy.unique(y_train, return_counts=True))
    print(numpy.unique(y_test, return_counts=True))



    train_data_sort=[]
    for i in range(len(y_train)):
        train_data_sort.append({'data':x_train[i],'label':y_train[i]})
        # print(train_data_sort[i])
    train_data_sort.sort(key=lambda k:(k.get('label')))

    x_data=[]
    x_label=[]
    for i in train_data_sort:
        x_data.append(i.get('data'))
        x_label.append(i.get('label'))

    ss = StandardScaler()
    x_data = ss.fit_transform(x_data)
    x_test = ss.transform(x_test)
    n_rules = rule



    if args.init == 'rpi':
        Cs, Vs = RPI(x_train, n_rules)
    elif args.init == 'kmean':
        Cs, Vs = kmean_init(x_train, n_rules)
    elif args.init == 'fcm':
        Cs, Vs = fcm_init(x_train, n_rules)
    else:
        exit()



    train_data = (x_data, x_label)
    test_data = (x_test, y_test)

    TMus=[]
    TMrs=[]
    TMas=[]
    AMrs=[]
    AM_rs=[]
    AM_fs=[]
    AMu_rs=[]
    AMu_fs=[]
    AMa_rs=[]
    AMa_fs=[]
    AMe2_rs=[]
    AMe2_fs=[]
    AMf1_rs=[]
    AMf1_fs=[]
    num_deletes=[]
    consistences=[]
    for i in range(len(fnums)):
        model = ClsTSK(x_train.shape[1], n_rules, n_classes, init_centers=Cs, init_Vs=Vs, bn=args.bn)
        model.load_state_dict(torch.load('diff_split/ckpt/{}_{}r_{}.pkl'.format(src,rule,acc)))

        classes = list(range(0, n_classes))
        random.seed()
        # classes_to_forget = random.sample(classes, fnums[i])
        classes_to_forget=[0]
        # classes_to_forget=[1,2]
        # classes_to_forget=[0,1,2]
        print("forget {}/{}".format(fnums[i], n_classes))
        print("forget classes: {}".format(classes_to_forget))
        model1 = copy.deepcopy(model)
        # print('******************M model rule num: {}'.format(model.n_rules))
        # print('******************M model classes  num: {}'.format(model.n_classes))

        AM_r,AM_f,TMu,AMu_r,AMu_f,AMe2_r, AMe2_f, AMf1_r, AMf1_f,num_delete,new_n_rule,mf2=EMNFU(model,train_data,test_data,512,n_classes,classes_to_forget,lrf1,lrf2,sample,Theta1,Theta2)
        # print('AMu_r: {}  A\':{}-----------------------------'.format(AMu_r,A))
        print('-------------------------------')

        # print('******************Mu model rule num: {}'.format(model.n_rules))
        # print('******************Mu model classes  num: {}'.format(model.n_classes))

        # AMr, TMr ,mr= retrainM(src,n_rules,classes_to_forget)
        # print('AMr: {} '.format(AMr))
        TMus.append(TMu)
        # AMrs.append(AMr)
        # TMrs.append(TMr)
        AM_rs.append(AM_r)
        AM_fs.append(AM_f)
        AMu_rs.append(AMu_r)

        AMu_fs.append(AMu_f)

        TMa,AMa_r,AMa_f = AmnesiscML(model1,train_data,test_data,n_classes,512,classes_to_forget)
        TMas.append(TMa)
        AMa_rs.append(AMa_r)
        AMa_fs.append(AMa_f)
        AMe2_rs.append(AMe2_r)
        AMe2_fs.append(AMe2_f)
        AMf1_rs.append(AMf1_r)
        AMf1_fs.append(AMf1_f)
        num_deletes.append(num_delete)

        # consistences.append(Consistence(mf2,mr,test_data))
    TMu=np.mean(TMus)
    TMr=np.mean(TMrs)
    TMa=np.mean(TMas)
    AMr=np.mean(AMrs)
    AMu_r=np.mean(AMu_rs)
    AMu_f=np.mean(AMu_fs)
    AMa_r=np.mean(AMa_rs)
    AMa_f=np.mean(AMa_fs)
    AM_r=np.mean(AM_rs)
    AM_f=np.mean(AM_fs)
    AMe2_r=np.mean(AMe2_rs)
    AMe2_f=np.mean(AMe2_fs)
    AMf1_r=np.mean(AMf1_rs)
    AMf1_f=np.mean(AMf1_fs)
    num_delete=np.mean(num_deletes)
    consistence=np.mean(consistences)

    print('-----------------Mu VS Mr--------------------')
    print(TMus)
    print(TMrs)
    print(TMas)
    print('TMu:{}  TMr: {} TMa: {}  AM_r: {}  AM_f: {}  AMe2r: {}  AMe2_f: {} num_delete:{}  AMf1_r: {}  AMf1_f: {}  AMr: {} consistence:{} AMu_r:{}  AMu_f:{}  AMa_r:{}   AMa_f:{}'.format(TMu,TMr,TMa,AM_r,AM_f,AMe2_r,AMe2_f,num_delete,AMf1_r,AMf1_f,AMr,consistence,AMu_r,AMu_f,AMa_r,AMa_f))

def Consistence(mf,mr,test_data):
    data = []
    label = []
    all_num=len(test_data[1])
    for i in range(all_num):
        data.append(test_data[0][i])
        label.append(test_data[1][i])

    D = (fast_list2arr(data), fast_list2arr(label))
    D_loader= data_loader(D, True, batch_size=512, shuffle=False)
    mf.eval()
    mr.eval()
    consistence_num = 0
    acc1=0
    acc2=0
    for s, (inputs, labels) in enumerate(D_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        out1 = mf(inputs)
        out2 = mr(inputs)
        # print(out)
        pred1 = t.argmax(out1, dim=1)
        pred2 = t.argmax(out2, dim=1)
        pred2=pred2+1

        consistence_num += t.sum(pred1 == pred2).item()
        acc1 += t.sum(pred1 == labels).item()
        acc2 += t.sum(pred2 == labels).item()
        # print(pred1)
        # print(labels)
    # print(acc1/all_num)
    # print(acc2/all_num)
    print(consistence_num/all_num)

    return consistence_num/all_num

def retrainM(src,new_n_rule,classes_to_forget):

    acc,runtime,model=trainModel.trainModel2(args,src,'Mr',new_n_rule,classes_to_forget)

    return acc,runtime,model
def ErrorMaxNoise(model,classes_to_forget,batch_size, shape):
    noises = {}
    for cls in classes_to_forget:
        # print("Optiming loss for class {}".format(cls))
        noises[cls] = Noise(batch_size, shape).cuda()
        # print(noises[cls]())
        opt = torch.optim.Adam(noises[cls].parameters(), lr=0.1)

        num_epochs = 5
        num_steps = 8
        class_label = cls
        for epoch in range(num_epochs):
            total_loss = []
            for batch in range(num_steps):
                inputs = noises[cls].noise
                labels = torch.zeros(batch_size).cuda() + class_label
                outputs = model(inputs)

                loss = -F.cross_entropy(outputs, labels.long()) + 0.1 * torch.mean(torch.sum(torch.square(inputs), [1]))
                # loss = F.cross_entropy(outputs, labels.long())
                # loss = -F.nll_loss(outputs, labels.long()) + 0.1 * torch.mean(torch.sum(torch.square(inputs), [1]))
                # 清空梯度
                opt.zero_grad()
                # 计算梯度
                loss.backward()
                # 更新参数
                opt.step()
                total_loss.append(loss.cpu().detach().numpy())
            # print("Loss: {}".format(np.mean(total_loss)))
        # print(noises[cls].noise)
    return noises

def ErrorMinNoise(model,classes_to_forget,batch_size, shape):
    noises = {}
    for cls in classes_to_forget:
        # print("Optiming loss for class {}".format(cls))
        noises[cls] = Noise(batch_size, shape).cuda()
        # print(noises[cls]())
        opt = torch.optim.Adam(noises[cls].parameters(), lr=0.1)

        num_epochs = 5
        num_steps = 8
        class_label = cls
        for epoch in range(num_epochs):
            total_loss = []
            for batch in range(num_steps):
                inputs = noises[cls].noise
                labels = torch.zeros(batch_size).cuda() + class_label
                outputs = model(inputs)

                # loss = -F.cross_entropy(outputs, labels.long()) + 0.1 * torch.mean(torch.sum(torch.square(inputs), [1]))
                loss = F.cross_entropy(outputs, labels.long())
                # loss = -F.nll_loss(outputs, labels.long()) + 0.1 * torch.mean(torch.sum(torch.square(inputs), [1]))
                # 清空梯度
                opt.zero_grad()
                # 计算梯度
                loss.backward()
                # 更新参数
                opt.step()
                total_loss.append(loss.cpu().detach().numpy())
            # print("Loss: {}".format(np.mean(total_loss)))
        # print(noises[cls].noise)
    return noises

def NUDrsub(min_noises,max_noises,Drsub,n_classes,classes_to_forget):

    num_batches = 20
    # # print('-1')
    # min_noisy_data = []
    # min_noisy_label = []
    max_noisy_data = []
    max_noisy_label = []
    for cls in classes_to_forget:
        for i in range(num_batches):
            # batch = min_noises[cls]().cpu().detach()
            batch2 = max_noises[cls]().cpu().detach()
            j=random.randint(0, len(batch2)-1)
            # print(batch[j])
            # print(len(batch[j]))
            # noisy_data.append((batch[j], random.choice(Fr)))
            # min_noisy_data.append((batch[j], cls))
            # min_noisy_data.append(np.asarray(batch[j]))
            # min_noisy_label.append(cls)
            max_noisy_data.append(np.asarray(batch2[j]))
            max_noisy_label.append(cls)
            # noisy_data.append((batch[j], 0))
    # min_noisy_data = fast_list2arr(min_noisy_data)
    # min_noisy_label = np.asarray(min_noisy_label)
    # min_noisy_loader = data_loader((min_noisy_data, min_noisy_label), True, batch_size=512)
    max_noisy_data = fast_list2arr(max_noisy_data)
    max_noisy_label = np.asarray(max_noisy_label)
    max_noisy_loader = data_loader((max_noisy_data, max_noisy_label), True, batch_size=512)

    Drsub_data = Drsub[0]
    Drsub_label = Drsub[1]

    Fr = list(range(n_classes))
    for cls in classes_to_forget:
        Fr.remove(cls)
    max_noisy_data = []
    for cls in classes_to_forget:
        for i in range(num_batches):
            batch = max_noises[cls]().cpu().detach()
            j = random.randint(0, len(batch) - 1)
            # print(batch[j])
            # print(len(batch[j]))
            max_noisy_data.append((batch[j], random.choice(Fr)))
    # print(type(Drsub_data))
    for i in range(len(max_noisy_data)):
        Drsub_data=np.append(Drsub_data, [max_noisy_data[i][0]],axis=0)
        Drsub_label=np.append(Drsub_label, [max_noisy_data[i][1]],axis=0)


    noisy_data = (Drsub_data, Drsub_label)
    noisyDrsub_loader = data_loader(noisy_data, True, batch_size=512)
    # d_rsub_loader=data_loader((Drsub_data,Drsub_label), True, batch_size=512)
    # return noisy_loader
    # return min_noisy_loader,noisyDrsub_loader,d_rsub_loader
    return noisyDrsub_loader,max_noisy_loader

def Unlearning(lrf,epoch,noisy_loader,model,):
    optimizer = torch.optim.Adam(model.parameters(), lr=lrf)

    for epoch in range(epoch):
        model.train(True)
        for i, data in enumerate(noisy_loader):
            inputs, labels = data
            # print(labels)
            inputs, labels = inputs.cuda(), torch.tensor(labels).cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)

            loss.backward()
            optimizer.step()
    return model
def EMNFU(model,train_data,valid_data,batch_size,n_classes,classes_to_forget,lrf1,lrf2,sample,Theta1,Theta2):

    r_data=[]
    r_label=[]
    f_data=[]
    f_label = []

    # r0_data=[]
    # r0_label=[]
    # for i in range(len(valid_data[1])):
    #     if valid_data[1][i]==0:
    #         r0_data.append(valid_data[0][i])
    #         # print(valid_data[0][i])
    #         r0_label.append(valid_data[1][i])
    # Dr0 = (fast_list2arr(r0_data), fast_list2arr(r0_label))
    # print('r0:')
    # AMu_r0, AMu_f0 = checkAcc(model, Dr0, Dr0)

    for i in range(len(valid_data[1])):
        if valid_data[1][i] not in classes_to_forget:
            r_data.append(valid_data[0][i])
            # print(valid_data[0][i])
            r_label.append(valid_data[1][i])
        else:
            f_data.append(valid_data[0][i])
            f_label.append(valid_data[1][i])
    Dr=(fast_list2arr(r_data),fast_list2arr(r_label))
    Df=(fast_list2arr(f_data),fast_list2arr(f_label))

    rt_data=[]
    rt_label=[]
    for i in range(len(train_data[1])):
        if train_data[1][i] not in classes_to_forget:
            rt_data.append(train_data[0][i])
            rt_label.append(train_data[1][i])

    if sample==1:
        Drsub=(fast_list2arr(rt_data),fast_list2arr(rt_label))
    else:
        # Drsub  随机取样
        _, nums = numpy.unique(train_data[1], return_counts=True)
        nums2 = copy.deepcopy(nums)
        # print(nums)
        for i in range(len(nums2)):
            if i > 0:
                nums2[i] = nums2[i] + nums2[i - 1]
        # print(nums2)
        sump_idx = []
        for i in range(len(nums)):
            N = range(nums[i])
            sump_idx.append(random.sample(N, int(nums[i] * sample)))
        # print(sump_idx)
        sample_index = []
        for i in range(len(sump_idx)):
            if i not in classes_to_forget:
                # print(i)
                for j in sump_idx[i]:
                    if i == 0:
                        sample_index.append(j - 1)
                    else:
                        sample_index.append(nums2[i - 1] + j - 1)
        # print(sample_index)
        retain_samples_data = []
        retain_samples_label = []
        for i in sample_index:
            retain_samples_data.append(train_data[0][i])
            retain_samples_label.append(train_data[1][i])
            # print(train_data[1][i],end='')
        Drsub = (fast_list2arr(retain_samples_data), fast_list2arr(retain_samples_label))
        # print(len(sample_index))
        # print(len(Drsub[1]))
        # print(retain_samples_label)

    shape=len(r_data[0])
    # print(shape)
    #若为二维数据，则为shape 不能以len代替

    AMr,AMf=checkAcc(model, Dr, Df)
    print("_______________________________________________")
    # print("|train noises|")


    # min_noises=ErrorMinNoise(model,classes_to_forget,batch_size,shape)
    min_noises=[]
    # startMu = time.perf_counter()
    max_noises=ErrorMaxNoise(model,classes_to_forget,batch_size,shape)
    noisyDrsub_loader,max_noisy_loader=NUDrsub(min_noises,max_noises,Drsub,n_classes,classes_to_forget)
    model1= copy.deepcopy(model)

    model1 = Unlearning(lrf1,1, max_noisy_loader,model1)
    AMe2_r, AMe2_f = checkAcc(model1, Dr, Df)

    mf1,delete_num=delete_rule_about_Df(model,model1,Theta1,Theta2)

    AMf1_r, AMf1_f = checkAcc(mf1, Dr, Df)

    startMu = time.perf_counter()
    mf2 = Unlearning(lrf2, 2, noisyDrsub_loader, mf1)
    endMu = time.perf_counter()
    TMu = endMu - startMu






    # print(model1)
    # print(model1.Cons)
    # print(model1.Bias)
    # print(model1.Cs)
    # print(model1.Vs)


    # impair
    print("_______________________________________________")
    print("|forget|")
    # model2 = copy.deepcopy(model)
    # model3 = copy.deepcopy(model)
    # model2=Unlearning(lrf,2, d_rsub_loader,model2)
    # model3=Unlearning(lrf,2, d_rsub_loader,model3)
    #
    # A1,_=checkAcc(model1, Dr, Df)
    # A2,_=checkAcc(model2, Dr, Df)
    # A3,_=checkAcc(model3, Dr, Df)




    #参数取平均
    # models=[model1,model2,model3]
    # worker_state_dict = [x.state_dict() for x in models]
    # weight_keys = list(worker_state_dict[0].keys())
    # fed_state_dict = collections.OrderedDict()
    # for key in weight_keys:
    #     key_sum = 0
    #     for i in range(len(models)):
    #         key_sum = key_sum + worker_state_dict[i][key]
    #     fed_state_dict[key] = key_sum / len(models)
    # ### update fed weights to fl model
    # model.load_state_dict(fed_state_dict)


    # t.save(model.state_dict(), 'diff_split/ckpt/{}_{}.pkl'.format(src, 'Mf'))

    # print('')
    # print("average acc")
    AMu_r,AMu_f=checkAcc(mf2, Dr, Df)

    # print('r0:')
    # AMu_r0, AMu_f0 = checkAcc(model, Dr0, Dr0)
    return AMr,AMf,TMu,AMu_r,AMu_f,AMe2_r,AMe2_f,AMf1_r,AMf1_f,delete_num,mf2.n_rules,mf2

def AmnesiscML(model,train_data,valid_data,numofClass,batch_size,classes_to_forget):

    retainlabels = list(range(numofClass))
    for i in classes_to_forget:
        retainlabels.remove(i)

    r_data = []
    r_label = []
    f_data = []
    f_label = []
    unlearning_data=[]
    unlearning_label=[]
    for i in range(len(valid_data[1])):
        if valid_data[1][i] not in classes_to_forget:
            r_data.append(valid_data[0][i])
            r_label.append(valid_data[1][i])
        else:
            f_data.append(valid_data[0][i])
            f_label.append(valid_data[1][i])

    for i in range(len(train_data[1])):
        if train_data[1][i] not in classes_to_forget:
            unlearning_data.append(train_data[0][i])
            unlearning_label.append(train_data[1][i])
        else:
            unlearning_data.append(train_data[0][i])
            # print(random.choice(retainlabels))
            unlearning_label.append(random.choice(retainlabels))



    Dr = (fast_list2arr(r_data), fast_list2arr(r_label))
    Df = (fast_list2arr(f_data), fast_list2arr(f_label))
    Dunlearning=(fast_list2arr(unlearning_data), np.asarray(unlearning_label))

    Dunlearning_loader = data_loader(Dunlearning, True, batch_size=512)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    startMu = time.perf_counter()

    for epoch in range(3):
        model.train(True)
        running_loss = 0.0
        running_acc = 0

        # print(noisy_loader)
        for i, data in enumerate(Dunlearning_loader):
            # print(i)
            # print(data)
            inputs, labels = data
            inputs, labels = inputs.cuda(), torch.tensor(labels).cuda()
            model=model.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item() * inputs.size(0)
            out = torch.argmax(outputs.detach(), dim=1)
            assert out.shape == labels.shape
            running_acc += (labels == out).sum().item()
    print('AmnesiscML result')
    AMa_r,AMa_f = checkAcc(model, Dr, Df)
    endMu = time.perf_counter()
    TMa = endMu - startMu
    return TMa,AMa_r,AMa_f
def checkAcc(model, Dr,Df):
    # print(model.Vs)

    Train = ClassModelTrain(
        model=model, train_data=Dr,
        test_data=Df, n_classes=1,
        args=args, save_path='', optim_type=args.optim_type
    )
    ar,af=Train.ArandAf()
    print("Ar: {:.2f}".format(ar * 100),end='  ')
    print("Af: {:.2f}".format(af * 100))
    return ar * 100,af*100

#arraylist to array
def fast_list2arr(data, offset=None, dtype=None):
    """
    Convert a list of numpy arrays with the same size to a large numpy array.
    This is way more efficient than directly using numpy.array()
    See
        https://github.com/obspy/obspy/wiki/Known-Python-Issues
    :param data: [numpy.array]
    :param offset: array to be subtracted from the each array.
    :param dtype: data type
    :return: numpy.array
    """
    num = len(data)
    out_data = np.empty((num,) + data[0].shape, dtype=dtype if dtype else data[0].dtype)
    for i in range(num):
        out_data[i] = data[i] - offset if offset else data[i]
    return out_data

def data_loader(data, longlabel=False, batch_size=512, shuffle=True):
    t_data = []
    for n in range(len(data)):
        t_data.append(t.from_numpy(data[n]).float())
    if longlabel:
        t_data[-1] = t_data[-1].long()

    tenDataset = Data.TensorDataset(*t_data)
    return Data.DataLoader(
        dataset=tenDataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        # drop_last=True
    )




def analysis_out_rule(t,rule):
    t=torch.reshape(t,[rule,-1])
    a=torch.sum(torch.abs(t),dim=1)
    # print(a)
    return a


def delete_rule(model,Theta):
    # Cons=model.Cons
    a=analysis_out_rule(model.Cons,model.n_rules)
    print(a)
    deleta_index=[]
    for i in range(len(a)):
        if a[i]>Theta:
            deleta_index.insert(0,i)

    deleta_num=len(deleta_index)
    if deleta_num==0:
        print("no rule to delete")
        return model
    else:
        print('delete {} rules'.format(deleta_num))
        print(deleta_index)
        for i in deleta_index:
            a=model.Cons[0:i]
            b=model.Cons[i+1:]
            model.Cons=nn.Parameter(torch.cat((a,b),dim=0))
            a=model.Bias[:,:i]
            b=model.Bias[:,i+1:]
            model.Bias = nn.Parameter(torch.cat((a, b), dim=1))
            a = model.Cs[:, :i]
            b = model.Cs[:, i + 1:]
            model.Cs = nn.Parameter(torch.cat((a, b), dim=1))
            a = model.Vs[:, :i]
            b = model.Vs[:, i + 1:]
            model.Vs = nn.Parameter(torch.cat((a, b), dim=1))
        model.n_rules=model.n_rules-deleta_num
    return model

def delete_rule_about_Df(model,modelf,Theta1,Theta2):
    Cons_m=torch.reshape(model.Cons,[model.n_rules,-1])
    Cons_mf=torch.reshape(modelf.Cons,[modelf.n_rules,-1])
    # print(Cons_m)
    # print(Cons_mf)
    # print(torch.abs(Cons_mf-Cons_m)/torch.abs(Cons_m))
    deleta_index=[]
    changeCons=torch.sum(torch.abs(Cons_mf-Cons_m)/torch.abs(Cons_m),dim=1)

    print(changeCons)
    for i in range(len(changeCons)):
        if Theta1<=changeCons[i]<Theta2:
        # if changeCons[i]<Theta1:
        # if changeCons[i]>Theta2:
            deleta_index.insert(0,i)

    deleta_num=len(deleta_index)
    if deleta_num==0:
        print("no rule to delete")
        return model,0
    else:
        print('delete {} rules'.format(deleta_num))
        print(deleta_index)
        for i in deleta_index:
            a=model.Cons[0:i]
            b=model.Cons[i+1:]
            model.Cons=nn.Parameter(torch.cat((a,b),dim=0))
            a=model.Bias[:,:i]
            b=model.Bias[:,i+1:]
            model.Bias = nn.Parameter(torch.cat((a, b), dim=1))
            a = model.Cs[:, :i]
            b = model.Cs[:, i + 1:]
            model.Cs = nn.Parameter(torch.cat((a, b), dim=1))
            a = model.Vs[:, :i]
            b = model.Vs[:, i + 1:]
            model.Vs = nn.Parameter(torch.cat((a, b), dim=1))
        model.n_rules=model.n_rules-deleta_num
    return model,deleta_num

if __name__ == '__main__':




    # prepare('HandwrittenDigits', [1,1,1,1,1,1,1,1,1,1], 39, 95, 0.1,0.1, 1,25000,30000)
    # prepare('PenBasedHandwrittenDigits',[1,1,1,1,1,1,1,1,1,1],26,99,0.23,0.23,1,20000,25000)
    # prepare('LetterRecognition',[1,1,1,1,1,1,1,1,1,1],44,88,0.2,0.04,1,0,0)
    # prepare('RoomOccupancy', [1,1,1,1,1,1,1,1,1,1], 6, 99, 0.1,0.1, 1,0,0)
    # prepare('DryBean', [1,1,1,1,1,1,1,1,1,1], 18, 92, 0.1,0.1, 1,3000,4000)
    # prepare('Nursery', [1,1,1,1,1,1,1,1,1,1], 8, 96, 0.1,0.1, 1,0,200)
    # prepare('Satellite',[1,1,1,1,1,1,1,1,1,1],25,90,0.1,0.15,1,8000,15000)
    # prepare('Connect-4',[1,1,1,1,1,1,1,1,1,1],62,78,0.1,0.1,1,100000,500000)
    # prepare('Shuttle',[1,1,1,1,1,1,1,1,1,1],11,99,0.1,0.1,1,1000,3000)
    # prepare('Covertype',[1,1,1,1,1,1,1,1,1,1],68,78,0.1,0.13,0.1,1000,100000)
    
    #[0]
    #prepare('PenBasedHandwrittenDigits', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 26, 99, 0.23, 0.23, 1, 20000, 25000)
    #[1,3]
    # prepare('PenBasedHandwrittenDigits', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 26, 99, 0.23, 0.23, 1, 20000, 30000)
    #[2,4,6,8]
    # prepare('PenBasedHandwrittenDigits', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 26, 99, 0.09, 0.23, 1, 20000, 200000)
    #[0,1,2,3,4,5,6,7]
    prepare('PenBasedHandwrittenDigits', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 26, 99, 0.09, 0.23, 1, 15000, 200000)



    # prepare('Connect-4', [1, 1, 1, 1, 1, 1], 62, 78, 0.1, 1,40)
    # prepare('Covertype', [1], 68, 78, 0.13, 0.1,100000,500000)


    # prepare('Nursery',[3],6,91,0.1,1)
    #
    # compareMfM()
























