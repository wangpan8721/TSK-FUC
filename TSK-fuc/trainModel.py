import argparse
import os
import copy
import numpy
import numpy as np
import torchvision.transforms as tt
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import scipy.io as sp
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import random
from lib.inits import *
from lib.models import *
from lib.tuning_train import *
import torch.nn.functional as F
import torch.utils.data as Data
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import  tensorflow as  tf
from PIL import Image
from keras.datasets import mnist
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time






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
    parser.add_argument('--epochs', default=400, type=int, help='total training epochs')
    parser.add_argument('--patience', default=20, type=int, help='training patience for early stopping')
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
data_root = 'data/'

def Yeastrun(flag, tail):
    args.flag = flag
    save_path = 'diff_split/ckpt/{}_{}_{}.pkl'.format(args.data, flag, tail)

    f = np.load(os.path.join(data_root, args.data + '.npz'))
    if flag == 0:
        print('Loading {} data, saving to {}'.format(args.data, save_path))
    data = f['con_data'].astype('float')
    label = f['label']
    n_classes = len(np.unique(label))
    train_idx, test_idx = f['trains'][flag], f['tests'][flag]

    x_train, y_train = data[train_idx], label[train_idx]
    x_test, y_test = data[test_idx], label[test_idx]

    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    n_rules = args.n_rules

    if args.init == 'rpi':
        Cs, Vs = RPI(x_train, n_rules)
    elif args.init == 'kmean':
        Cs, Vs = kmean_init(x_train, n_rules)
    elif args.init == 'fcm':
        Cs, Vs = fcm_init(x_train, n_rules)
    else:
        exit()

    model = ClsTSK(x_train.shape[1], n_rules, n_classes, init_centers=Cs, init_Vs=Vs, bn=args.bn)

    Train = ClassModelTrain(
        model=model, train_data=(x_train, y_train),
        test_data=(x_test, y_test), n_classes=n_classes,
        args=args, save_path=save_path, optim_type=args.optim_type
    )
    best_test_bca, best_test_acc = Train.train()
    print('[FLAG {:2d}] ACC: {:.4f}, BCA: {:.4f}'.format(flag, best_test_acc, best_test_bca))


    return best_test_acc, best_test_bca

#数组列表转数组
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


def loaddata(src):
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







def run(tail,src,type):
    # save_path = 'diff_split/ckpt2/{}_{}.pkl'.format(src,  tail)

    save_path1 = 'diff_split/ckpt2/{}{}_{}.pkl'.format(src, type,tail)
    # x,y=loaddata(src)
    x,y=loaddata(src)
    n_classes = len(np.unique(y))
    print(n_classes)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    n_rules = args.n_rules

    if args.init == 'rpi':
        Cs, Vs = RPI(x_train, n_rules)
    elif args.init == 'kmean':
        Cs, Vs = kmean_init(x_train, n_rules)
    elif args.init == 'fcm':
        Cs, Vs = fcm_init(x_train, n_rules)
    else:
        exit()

    runtimes=[]
    accs=[]
    bcas=[]
    for i in range(1):
        start=time.perf_counter()
        model = ClsTSK(x_train.shape[1], n_rules, n_classes, init_centers=Cs, init_Vs=Vs, bn=args.bn)
        Train = ClassModelTrain(
            model=model, train_data=(x_train, y_train),
            test_data=(x_test, y_test), n_classes=n_classes,
            args=args, save_path=save_path1, optim_type=args.optim_type)

        best_test_bca, best_test_acc = Train.train()
        end=time.perf_counter()
        runtime=(end-start)

        runtimes.append(runtime)
        accs.append(best_test_acc)
        bcas.append(best_test_bca)

    runtime=np.mean(runtimes)
    acc=np.mean(accs)
    bca=np.mean(bcas)
    print(accs)
    print(runtimes)
    print('ACC: {:.4f}, RunTime: {:.4f}'.format( acc, runtime))


    return acc, bca,runtime

def run2Acc(tail,src,type,new_n_rule,classes_to_forget):
    # save_path = 'diff_split/ckpt2/{}_{}.pkl'.format(src,  tail)

    save_path1 = 'diff_split/ckpt/{}{}_{}.pkl'.format(src, type,tail)
    # x,y=loaddata(src)
    x,y=loaddata(src)

    x1=[]
    y1=[]

    for i in range(len(y)):
        if y[i] not in classes_to_forget:
            x1.append(x[i])
            y1.append(y[i])
            #一致性只遗忘0类时
            # y1.append(y[i]-1)

    x=fast_list2arr(x1)
    y=np.asarray(y1)

    #默认遗忘第0类，其他标签-1
    le = LabelEncoder()
    le = le.fit(np.unique(y))
    y = le.transform(y)

    n_classes = len(np.unique(y))
    print(n_classes)



    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    n_rules = args.n_rules


    args.n_rules = new_n_rule



    runtimes=[]
    accs=[]
    bcas=[]

    if args.init == 'rpi':
        Cs, Vs = RPI(x_train, args.n_rules)
    elif args.init == 'kmean':
        Cs, Vs = kmean_init(x_train, args.n_rules)
    elif args.init == 'fcm':
        Cs, Vs = fcm_init(x_train, args.n_rules)
    else:
        exit()
    model = ClsTSK(x_train.shape[1], args.n_rules, n_classes, init_centers=Cs, init_Vs=Vs, bn=args.bn)
    Train = ClassModelTrain(
        model=model, train_data=(x_train, y_train),
        test_data=(x_test, y_test), n_classes=n_classes,
        args=args, save_path=save_path1, optim_type=args.optim_type)

    start = time.perf_counter()
    best_test_bca, best_test_acc ,mr= Train.train2Acc()
    end=time.perf_counter()
    runtime=(end-start)

    runtimes.append(runtime)
    accs.append(best_test_acc)
    bcas.append(best_test_bca)

    runtime=np.mean(runtimes)
    acc=np.mean(accs)
    bca=np.mean(bcas)

    print(accs)
    print(runtimes)
    print('ACC: {:.4f}, RunTime: {:.4f}'.format( acc, runtime))

    # print(model)
    # print(model.Cons.shape)
    # print(model.Bias.shape)
    # print(model.Cs.shape)
    # print(model.Vs.shape)
    # print('******************Mr model rule num: {}'.format(model.n_rules))
    # print('******************Mr model classes  num: {}'.format(model.n_classes))

    return acc, bca,runtime,mr

def runGridSearch(src):
    x, y = loaddata(src)
    n_classes = len(np.unique(y))
    print(n_classes)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)



    # set_seed(10)
    param_grid = {'n_rules': list(np.linspace(2, 100, 50, dtype=int)),}
    MAX_EVALS = 40

    best_acc = 0
    best_hyperparams = {}

    # for i in range(MAX_EVALS):
    for i in range(1):
        random.seed(i)
        random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        args.n_rules = random_params['n_rules']

        # n_rules = args.n_rules
        args.n_rules=n_rules = 26
        if args.init == 'rpi':
            Cs, Vs = RPI(x_train, n_rules)
        elif args.init == 'kmean':
            Cs, Vs = kmean_init(x_train, n_rules)
        elif args.init == 'fcm':
            Cs, Vs = fcm_init(x_train, n_rules)
        else:
            exit()


        print('{}/{}'.format(i, MAX_EVALS))
        print(random_params)

        save_path = 'diff_split/ckpt/{}_{}r_.pkl'.format(src, args.n_rules)

        model = ClsTSK(x_train.shape[1], n_rules, n_classes, init_centers=Cs, init_Vs=Vs, bn=args.bn)
        Train = ClassModelTrain(
            model=model, train_data=(x_train, y_train),
            test_data=(x_test, y_test), n_classes=n_classes,
            args=args, save_path=save_path, optim_type=args.optim_type)
        vol_acc = Train.trainGridSearch()
        if vol_acc>best_acc:
            best_acc=vol_acc
            best_hyperparams=random_params

        print('acc ： {}/{}'.format(vol_acc,best_acc))

    print('best rules:{}'.format(best_hyperparams['n_rules']))

    return 0

def trainModel(src):
    # n_repeats = 1
    # hist = [1] * n_repeats
    # best_acc = [0] * n_repeats
    # best_bca = [0] * n_repeats
    tail = args.loss_type
    if args.weight_frs > 0:
        tail += '_ur'
    if not args.bn:
        tail += '_noBN'
    if args.init != 'kmean':
        tail += '_{}'.format(args.init)
    acc,_,runtime = run(tail,src,'')
    return acc,runtime

def trainModel2(args,src,type,new_n_rule,classes_to_forget):

    tail = args.loss_type
    if args.weight_frs > 0:
        tail += '_ur'
    if not args.bn:
        tail += '_noBN'
    if args.init != 'kmean':
        tail += '_{}'.format(args.init)
    acc,_,runtime,model = run2Acc(tail,src,type,new_n_rule,classes_to_forget)
    return acc,runtime,model


def test(src,rule,acc):
    x, y = loaddata(src)


    n_classes = len(np.unique(y))



    ss = StandardScaler()
    x= ss.fit_transform(x)
    n_rules = rule

    if args.init == 'rpi':
        Cs, Vs = RPI(x, n_rules)
    elif args.init == 'kmean':
        Cs, Vs = kmean_init(x, n_rules)
    elif args.init == 'fcm':
        Cs, Vs = fcm_init(x, n_rules)
    else:
        exit()

    model = ClsTSK(x.shape[1], n_rules, n_classes, init_centers=Cs, init_Vs=Vs, bn=args.bn)
    model.load_state_dict(torch.load('diff_split/ckpt/{}_{}r_{}.pkl'.format(src,rule,acc)))

    print(model)
    train_data = ([], [])
    test_data = (x, y)
    Train = ClassModelTrain(
        model=model, train_data=train_data,
        test_data=test_data, n_classes=1,
        args=args, save_path='', optim_type=args.optim_type
    )
    bca,acc=Train.test()
    print('{} acc: {}'.format(src,acc))
    return bca,acc




def testNueseryMf():
    x, y = loaddata('Nursery')
    x1 = []
    y1 = []
    for i in range(len(y)):
        if y[i] not in [0,1,2]:
            x1.append(x[i])
            y1.append(y[i])
    x = fast_list2arr(x1)
    y = np.asarray(y1)


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    print(numpy.unique(y_train,return_counts=True))
    print(numpy.unique(y_test,return_counts=True))
    if args.init == 'rpi':
        Cs, Vs = RPI(x_train, 6)
    elif args.init == 'kmean':
        Cs, Vs = kmean_init(x_train, 6)
    elif args.init == 'fcm':
        Cs, Vs = fcm_init(x_train, 6)
    else:
        exit()

    ss = StandardScaler()
    x = ss.fit_transform(x)
    x_test = ss.transform(x_test)



    model = ClsTSK(x_train.shape[1],6, 5, init_centers=Cs, init_Vs=Vs, bn=args.bn)
    model.load_state_dict(torch.load('diff_split/ckpt/Nursery_6r_Mf_99.pkl'))

    # for _,parm in enumerate(model.parameters()):
    #     print(parm)
    # return 0



    print(model)
    train_data = ([], [])
    test_data = (x, y)
    Train = ClassModelTrain(
        model=model, train_data=train_data,
        test_data=test_data, n_classes=1,
        args=args, save_path='', optim_type=args.optim_type
    )
    bca,acc=Train.test()
    print('Nursery Mf acc: {}'.format(acc))
    return bca,acc


def abc():
    a=list(range(0,5))
    print(a)



if __name__ == '__main__':
    np.random.seed(1447)
    t.manual_seed(1447)


    # runGridSearch('HandwrittenDigits')
    # test('HandwrittenDigits', 60,95)
    # runGridSearch('PenBasedHandwrittenDigits')
    # test('PenBasedHandwrittenDigits', 68,99)
    # runGridSearch('LetterRecognition')
    # test('LetterRecognition', 74,88)
    # runGridSearch('RoomOccupancy')
    # test('RoomOccupancy', 14,99)
    # runGridSearch('DryBean')
    # test('DryBean', 62,92)
    # runGridSearch('Nursery')
    # test('Nursery', 32,96)
    # runGridSearch('Satellite')
    # test('Satellite', 42,90)
    # runGridSearch('Connect-4')
    # test('Connect-4', 62,78)
    # runGridSearch('Shuttle')
    # test('Shuttle', 88,99)
    # runGridSearch('Covertype')
    # test('Covertype', 20,74)



    # x,y=loaddata('Shuttle')
    # print(x[0])
    # print(len(x[0]))
    # print(len(y))
    # print(numpy.unique(y, return_counts=True))
    #Shuttle Satellite

    # runGridSearch('Nursery')
    # runGridSearch('PenBasedHandwrittenDigits')
    test('HandwrittenDigits', 39,'')
    # testNueseryMf()


    # abc()









