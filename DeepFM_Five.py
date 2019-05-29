"""
reference:
DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
https://arxiv.org/abs/1703.04247
"""
import torch.utils.data as Data
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
print(torch.cuda.is_available())
class DeepFM(nn.Module):
    def __init__(self,fieldsize=2,feasize=4677698,k=10):
        super(DeepFM, self).__init__()
        # build FM
        self.w=nn.Embedding(feasize,1)
        self.v=nn.Embedding(feasize,k)
        #build DNN
        hidden_layer = [fieldsize*k,16,16,16]
        self.dnn=torch.nn.Sequential()
        for i in range(len(hidden_layer)-1):
            self.dnn.add_module("Linear_"+str(i),nn.Linear(hidden_layer[i], hidden_layer[i+1]))
            self.dnn.add_module("Relu_"+str(i),nn.ReLU())
        self.dnn.add_module("Out",nn.Linear(hidden_layer[len(hidden_layer)-1],1))
    def forward(self, feature):
        batchsize=feature.shape[0]
        #2-order
        V=self.v(feature)
        res=0.5*((V.sum(dim=1)).pow(2)-(V.pow(2)).sum(1)).sum(dim=1)
        res=res.reshape(batchsize,1)
        #1-order
        W=self.w(torch.cuda.LongTensor(feature))
        res+=W.sum(dim=1)
        #dnn
        emb=V.reshape(batchsize,-1)
        res+=self.dnn(emb)
        #return(torch.sigmoid(res))
        return res


#data_input
Batchsize=3000
train_file_path="data/Electronics_train.csv"
test_file_path="data/Electronics_test.csv"
save_path="savemodel/smodel"
cols = ['user', 'item', 'rating']
train = pd.read_csv(train_file_path, delimiter=',', names=cols)
X_train=torch.LongTensor(train[['user','item']].values)
Y_train=torch.LongTensor(train['rating'].values)
train_dataset = Data.TensorDataset(X_train,Y_train)
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=Batchsize,
    shuffle=True,
    num_workers=2,
)
test = pd.read_csv(test_file_path, delimiter=',', names=cols)
X_test=torch.LongTensor(test[['user','item']].values)
Y_test=torch.LongTensor(test['rating'].values)
test_dataset = Data.TensorDataset(X_test,Y_test)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=Batchsize,
    shuffle=False,
    num_workers=2,
)


#train
models=[DeepFM().cuda()for i in range(5)]
print(models[0])
para=[]
for i in range(5):
    para.append({'params':models[i].parameters()})
optimizer = optim.Adam(para, lr=0.001)
criterion=nn.CrossEntropyLoss()
sm=nn.Softmax()
for epoch in range(50):
    loss_all = []
    for step, (batch_x, batch_y) in enumerate(train_loader):
        x=batch_x.cuda()
        y=batch_y.cuda()
        y-=1
        r = torch.Tensor().cuda()
        for i in range(5):
            resi = models[i](x)
            r = torch.cat((r, resi), dim=1)
        loss = criterion(r, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all.append(loss.item())
        if (step+1)%200==0:
            num_correct=0
            num_all=0
            for i in range(5):
                torch.save(models[i],save_path+str(i))
            for _, (tx, ty) in enumerate(test_loader):
                r = torch.Tensor().cuda()
                for i in range(5):
                    resi = models[i](tx.cuda())
                    r = torch.cat((r, resi), dim=1)
                r = sm(r)
                r = torch.argmax(r, dim=1)
                r+=1
                num_correct += (r == ty.cuda()).sum()
                num_all+=r.size(0)
            acc=float(num_correct) / num_all
            print('Epoch: ', epoch, '| Step: ', step+1, '| Loss: ',sum(loss_all)/len(loss_all),'| ACC: ',acc)
            loss_all=[]
