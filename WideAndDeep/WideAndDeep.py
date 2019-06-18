"""
reference:
Wide & Deep Learning for Recommender Systems
https://arxiv.org/pdf/1606.07792.pdf
"""
import torch.utils.data as Data
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
from sklearn import metrics
import sklearn
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
print(torch.cuda.is_available())
class WideDeep(nn.Module):
    def __init__(self,fieldsize=39,feasize=2325450,k=32):
        super(WideDeep, self).__init__()
        self.fieldsize=fieldsize
        self.feasize=feasize
        # build Wide (working
        # self.w2=nn.Embedding(feasize*feasize,1)
        self.w1=nn.Embedding(feasize,1)
        #build Deep
        self.em=nn.Embedding(feasize,k)
        hidden_layer = [fieldsize*k,1024,512,256]
        self.dnn=torch.nn.Sequential()
        for i in range(len(hidden_layer)-1):
            self.dnn.add_module("Linear_"+str(i),nn.Linear(hidden_layer[i], hidden_layer[i+1]))
            # self.dnn.add_module("Drop_"+str(i),nn.Dropout(0.5))
            self.dnn.add_module("Relu_"+str(i),nn.ReLU())
        self.dnn.add_module("Out",nn.Linear(hidden_layer[len(hidden_layer)-1],1))

    def forward(self, feature):
        batchsize=feature.shape[0]
        #deep
        emb = self.em(feature).reshape(batchsize, -1)
        outdnn=self.dnn(emb)
        res=outdnn
        #wide
        #1 order
        res+=self.w1(feature).sum(dim=1)
        #2 order
        # cross_feature=torch.Tensor(batchsize,self.fieldsize*(self.fieldsize-1)//2)
        # for i in range(batchsize):
        #     num=0
        #     for j in range(self.fieldsize):
        #         for k in range(j+1,self.fieldsize):
        #             cross_feature[i][num]=feature[i][j]*self.feasize+feature[i][k]
        #             num+=1
        # res+=self.w2(cross_feature).sum(dim=1)
        res=torch.sigmoid(res)
        res = res.reshape(-1)
        return res


#data_input
Batchsize=300
train_file_path="../data/train.csv"
test_file_path="../data/test.csv"
save_path="../savemodel/smodel"
labcols = ['lable']
feacols=[]
for i in range(13):
    feacols.append("I"+str(i))
for i in range(26):
    feacols.append("C" + str(i))
train = pd.read_csv(train_file_path, delimiter=',', names=labcols+feacols)
X_train=torch.LongTensor(train[feacols].values)
Y_train=torch.LongTensor(train['lable'].values)
train_dataset = Data.TensorDataset(X_train,Y_train)
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=Batchsize,
    shuffle=True,
    num_workers=1,
)
test = pd.read_csv(test_file_path, delimiter=',', names=labcols+feacols)
X_test=torch.LongTensor(test[feacols].values)
Y_test=torch.LongTensor(test['lable'].values)
test_dataset = Data.TensorDataset(X_test,Y_test)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=Batchsize,
    shuffle=False,
    num_workers=1,
)


#train
model=WideDeep().cuda()
print(model)
optimizer = optim.Adagrad(model.parameters(), lr=0.001)
criterion=nn.BCELoss()
for epoch in range(50):
    loss_all = []
    for step, (batch_x, batch_y) in enumerate(train_loader):
        x=batch_x.cuda()
        y=batch_y.cuda().float()
        r=model(x)
        loss = criterion(r, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all.append(loss.item())
        if (step+1)%20000==0:
            torch.save(model, save_path + 'ctr')
            print('Epoch: ', epoch, '| Step: ', step+1, '| Train_Loss: ',sum(loss_all)/len(loss_all))
            loss_all=[]
    #test
    num_correct = 0
    num_all = 0
    test_loss = 0
    auc_part=0
    y_true_all=np.ones(0)
    y_score_all=np.ones(0)
    for _, (tx, ty) in enumerate(test_loader):
        y_c = ty.cuda()
        r=model(tx.cuda())
        test_loss += criterion(r, y_c.float()).item() * r.size(0)
        y_true=ty.numpy()
        y_true_all=np.concatenate([y_true_all,y_true])
        y_score=r.cpu()
        y_score=y_score.detach().numpy()
        y_score_all=np.concatenate([y_score_all,y_score])
        r=(r>0.5).long()
        num_correct += (r == y_c).sum()
        num_all += r.size(0)
    acc = float(num_correct) / num_all
    auc=sklearn.metrics.roc_auc_score(y_true_all,y_score_all)
    print('Epoch: ', epoch, '| Test_Loss: ',test_loss / num_all, '| Test_ACC: ', acc,'| Test_AUC: ',auc)