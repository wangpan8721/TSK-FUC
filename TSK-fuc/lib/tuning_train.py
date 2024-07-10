import random
from keras.losses import mean_squared_error
from lib.torch_utils import get_loss_func, get_optim_func, data_loader
import torch as t
from time import time
import numpy as np
import os
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import itertools
import torch


def eval_acc(model, loader, cuda=False):
    model.eval()
    num_correct = 0
    num_data = 0
    for s, (inputs, labels) in enumerate(loader):
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        out = model(inputs)
        # print(out)
        pred = t.argmax(out, dim=1)
        num_correct += t.sum(pred == labels).item()
        num_data += labels.size(0)
        # print(labels)
    # print('num_data:{}'.format(num_data))
    # print('num_correct:{}'.format(num_correct))
    return num_correct / num_data


def eval_bca(model, loader, cuda=False):
    model.eval()
    outs, trues = [], []
    for s, (inputs, labels) in enumerate(loader):
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        out = model(inputs)
        pred = t.argmax(out, dim=1)
        outs.append(pred)
        trues.append(labels)
    return balanced_accuracy_score(
        t.cat(trues, dim=0).detach().cpu().numpy(),
        t.cat(outs, dim=0).detach().cpu().numpy()
    )


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return 0
class ClassModelTrain():
    def __init__(self, model, train_data, test_data=None,
                 n_classes=None, optim_type='adabound',
                 args=None, save_path='tmp.pkl'):
        if args is None:
            raise ValueError('Args can\'t be None')
        self.optim_func = get_optim_func(optim_type)
        self.loss_func = get_loss_func(args.loss_type, regression=False, n_classes=n_classes)
        self.save_path = save_path
        self.args = args
        if args.gpu:
            model.cuda()
        self.model = model
        self.train_data = train_data
        self.test_data = test_data

    def split_train_val(self, val_size, random_state=None):
        x_train, y_train = self.train_data
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=val_size, random_state=random_state
        )
        return x_train, y_train, x_val, y_val


    def train(self):
        if self.args.weight_frs > 0:
            range_ur = [0.1,  1, 10, 20, 50]
            bcas = np.zeros([len(range_ur)])
            stops = np.zeros([len(range_ur)])
            for i in range(self.args.repeats):
                for k, w_ur in enumerate(range_ur):
                    bca, _, pos = self._train_(self.args.weight_decay, w_ur=w_ur)
                    bcas[k] += bca
                    stops[k] += pos
            bcas /= self.args.repeats
            stops /= self.args.repeats
            best_idx = int(np.argmax(bcas))
            print('\nbest ur param: {}, best stop pos: {}'.format(
                range_ur[best_idx], stops[best_idx]
            ))
            bca, acc = self.train_final_model(self.args.weight_decay, range_ur[best_idx], int(round(stops[best_idx])))
            return bca, acc
        stops = 0
        for i in range(self.args.repeats):
            _, _, pos = self._train_(self.args.weight_decay, w_ur=0)
            stops += pos
        stops /= self.args.repeats
        print('\nbest stop pos: {}'.format(stops))
        bca, acc = self.train_final_model(self.args.weight_decay, 0, int(round(stops)))

        return bca, acc


    def trainGridSearch(self):
        best_acc = 0
        w_l2 = self.args.weight_decay
        w_ur = 0
        x_train, y_train = self.train_data
        self.model.rebuild_model(self.args.gpu)

        optim = self.optim_func(self.model.parameters(), lr=self.args.lr)
        tester = data_loader(self.test_data, True, batch_size=self.args.batch_size, shuffle=False)
        trainer = data_loader([x_train, y_train], True, batch_size=self.args.batch_size)
        for e in range(self.args.epochs):
            self.model.train()
            for s, (inputs, labels) in enumerate(trainer):
                if self.args.gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                out, frs = self.model(inputs, with_frs=True)
                # print(out)
                loss = self.loss_func(out, labels) + \
                       w_l2 * self.model.l2_loss() + \
                       w_ur * self.model.ur_loss(frs)
                optim.zero_grad()
                loss.backward()
                optim.step()

            val_acc = eval_acc(self.model, tester, self.args.gpu)

            if val_acc > best_acc:
                best_acc = val_acc
                t.save(self.model.state_dict(), self.save_path)
                count = 0
            else:
                count += 1
                if count > self.args.patience:
                    break
        return best_acc
    def retrainGridSearch(self):


        best_acc = 0
        w_l2 = self.args.weight_decay
        w_ur = 0
        x_train, y_train = self.train_data
        self.model.rebuild_model(self.args.gpu)

        optim = self.optim_func(self.model.parameters(), lr=self.args.lr)
        tester = data_loader(self.test_data, True, batch_size=self.args.batch_size, shuffle=False)
        trainer = data_loader([x_train, y_train], True, batch_size=self.args.batch_size)
        for e in range(self.args.epochs):
            self.model.train()
            for s, (inputs, labels) in enumerate(trainer):
                if self.args.gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                out, frs = self.model(inputs, with_frs=True)
                loss = self.loss_func(out, labels) + \
                       w_l2 * self.model.l2_loss() + \
                       w_ur * self.model.ur_loss(frs)
                optim.zero_grad()
                loss.backward()
                optim.step()

            val_acc = eval_acc(self.model, tester, self.args.gpu)

            if val_acc > best_acc:
                best_acc = val_acc
                # t.save(self.model.state_dict(), self.save_path)
                count = 0
            else:
                count += 1
                if count > self.args.patience:
                    break
            # print('{}  acc: {}/{}'.format(count,val_acc, best_acc))
        return best_acc

    def train2Acc(self):

        w_l2=self.args.weight_decay
        w_ur=0
        x_train, y_train = self.train_data
        self.model.rebuild_model(self.args.gpu)

        optim = self.optim_func(self.model.parameters(), lr=self.args.lr)
        tester = data_loader(self.test_data, True, batch_size=self.args.batch_size, shuffle=False)
        trainer = data_loader([x_train, y_train], True, batch_size=self.args.batch_size)
        best_acc = 0
        for e in range(self.args.epochs):
            self.model.train()
            for s, (inputs, labels) in enumerate(trainer):
                if self.args.gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                out, frs = self.model(inputs, with_frs=True)
                loss = self.loss_func(out, labels) + \
                       w_l2 * self.model.l2_loss() + \
                       w_ur * self.model.ur_loss(frs)
                optim.zero_grad()
                loss.backward()
                optim.step()
            best_test_acc = eval_acc(self.model, tester, self.args.gpu)
            print('\rbest acc:{} epoch: {}'.format(best_test_acc,e),end='')
            # if AMu_r>75:
            #     AMu_r=72
            # if best_test_acc*100>=AMu_r*xx:
            # # if best_test_acc*100>94:
            # #     t.save(self.model.state_dict(), self.save_path)
            #     break
            if best_test_acc > best_acc:
                best_acc = best_test_acc
                count = 0
            else:
                count += 1
                if count > self.args.patience:
                    break
        best_test_acc = eval_acc(self.model, tester, self.args.gpu)
        best_test_bca = eval_bca(self.model, tester, self.args.gpu)
        # t.save(self.model.state_dict(), self.save_path)

        return best_test_bca, best_test_acc,self.model


    def test(self):
        tester = data_loader(self.test_data, True, batch_size=self.args.batch_size, shuffle=False)
        acc = eval_acc(self.model, tester, self.args.gpu)
        bca = eval_bca(self.model, tester, self.args.gpu)
        return bca, acc
    def ArandAf(self):
        forget = data_loader(self.test_data, True, batch_size=self.args.batch_size, shuffle=False)
        retain = data_loader(self.train_data, True, batch_size=self.args.batch_size, shuffle=False)

        ar = eval_acc(self.model, retain, self.args.gpu)
        af = eval_acc(self.model, forget, self.args.gpu)
        return ar, af

    def train_final_model(self, w_l2, w_ur, epochs):
        x_train, y_train = self.train_data
        self.model.rebuild_model(self.args.gpu)

        optim = self.optim_func(self.model.parameters(), lr=self.args.lr)
        tester = data_loader(self.test_data, True, batch_size=self.args.batch_size, shuffle=False)
        trainer = data_loader([x_train, y_train], True, batch_size=self.args.batch_size)
        for e in range(epochs):
            self.model.train()
            for s, (inputs, labels) in enumerate(trainer):
                if self.args.gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                out, frs = self.model(inputs, with_frs=True)
                loss = self.loss_func(out, labels) + \
                       w_l2 * self.model.l2_loss() + \
                       w_ur * self.model.ur_loss(frs)
                optim.zero_grad()
                loss.backward()
                optim.step()
        t.save(self.model.state_dict(), self.save_path)
        best_test_acc = eval_acc(self.model, tester, self.args.gpu)
        best_test_bca = eval_bca(self.model, tester, self.args.gpu)
        return best_test_bca, best_test_acc


    def _train_(self, w_l2, w_ur, random_state=None, val_size=0.2):
        self.model.rebuild_model(self.args.gpu)
        x_train, y_train, x_val, y_val = self.split_train_val(val_size, random_state)
        x_train, y_train = self.train_data
        # print(x_train)
        optim = self.optim_func(self.model.parameters(), lr=self.args.lr)
        best_acc, count, best_test_acc, best_pos = 0, 0, 0, 0

        trainer = data_loader([x_train, y_train], True, batch_size=self.args.batch_size)
        valer = data_loader([x_val, y_val], True, batch_size=self.args.batch_size, shuffle=False)

        for e in range(self.args.epochs):
            self.model.train()
            start_t = time()
            for s, (inputs, labels) in enumerate(trainer):
                if self.args.gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                out, frs = self.model(inputs, with_frs=True)
                # print(out)

                loss = self.loss_func(out, labels) + \
                       w_l2 * self.model.l2_loss() + \
                       w_ur * self.model.ur_loss(frs)

                optim.zero_grad()
                loss.backward()
                optim.step()
            val_acc = eval_acc(self.model, valer, self.args.gpu)
            end_t = time()
            if val_acc > best_acc:
                best_acc = val_acc
                count = 0
                t.save(self.model.state_dict(), self.save_path + '.tmp')
                best_pos = e
            else:
                count += 1
                if count > self.args.patience:
                    break

            print('\r[TRAIN {:4d}] Val ACC: {:.4f}, Best Val ACC: {:.4f}, Time: {:.2f}s ,Count:{}  '.format(
                 e, val_acc, best_acc, end_t - start_t,count), end='')
        self.model.load_state_dict(t.load(self.save_path + '.tmp'))
        best_bca = eval_bca(self.model, valer, self.args.gpu)
        return best_bca, best_acc, best_pos


