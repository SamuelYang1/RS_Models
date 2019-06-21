import torch.utils.data as Data
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
from sklearn import metrics
import random
import sklearn
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
print(torch.cuda.is_available())
class DIN(nn.Module):
    def __init__(self,k=10):
        super(DIN, self).__init__()
        feasize=10000000
        self.user=nn.Embedding(138500,k)
        self.movie=nn.Embedding(131270,k)
        self.movie_cate = nn.Embedding(138500, k)
        self.user_rate = nn.Embedding(131270, k)
        #build DNN
        hidden_layer = [4*k,20,8]
        self.dnn=torch.nn.Sequential()
        for i in range(len(hidden_layer)-1):
            self.dnn.add_module("Linear_"+str(i),nn.Linear(hidden_layer[i], hidden_layer[i+1]))
            # self.dnn.add_module("Drop_"+str(i),nn.Dropout(0.5))
            self.dnn.add_module("Relu_"+str(i),nn.ReLU())
        self.dnn.add_module("Out",nn.Linear(hidden_layer[len(hidden_layer)-1],2))
        self.dnn.add_module("Softmax", nn.Softmax(dim=0))
    def forward(self, user_id,movie_id,movie_cate,user_rate):
        a=self.user(user_id).reshape(-1)
        b=self.movie(movie_id).reshape(-1)
        c=self.movie_cate(movie_cate).sum(dim=0)/movie_cate.shape[0]
        d=self.movie_cate(user_rate).sum(dim=0)/user_rate.shape[0]
        fea=torch.cat((a,b,c,d))
        res=self.dnn(fea)
        return res[1].reshape(1)
#data_input
Batchsize=300
train_file_path="../data/ml-20m/mini-train.txt"
test_file_path="../data/ml-20m/mini-test.txt"
save_path="../savemodel/smodel"
with open(train_file_path,"r")as ftrain,open(test_file_path,"r")as ftest:
    ss=ftrain.readlines()
    train_data=[eval(s) for s in ss]
    ss = ftest.readlines()
    test_data=[eval(s)for s in ss]
model=DIN().cuda()
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion=nn.BCELoss()
for epoch in range(50):
    loss_all = []
    num=0
    random.shuffle(train_data)
    for feature in train_data:
        num+=1
        user_id_t = torch.LongTensor([feature["user_id"]]).cuda()
        movie_id_t = torch.LongTensor([feature["movie_id"]]).cuda()
        movie_cate_t = torch.LongTensor(feature["movie_cate_id_list"]).cuda()
        user_rate_t = torch.LongTensor(feature["user_rated_movie_id_list"]).cuda()
        lable_t = torch.Tensor([feature["lable"]]).cuda()
        r = model(user_id_t, movie_id_t, movie_cate_t, user_rate_t)
        loss = criterion(r, lable_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all.append(loss.item())
        if num % 200000 == 0:
            torch.save(model, save_path + 'ctr')
            print('Epoch: ', epoch, '| Step: ', num, '| Train_Loss: ', sum(loss_all) / len(loss_all))
            loss_all = []
    #test
    num_correct = 0
    num_all = 0
    test_loss = 0
    auc_part = 0
    y_true_all = np.ones(0)
    y_score_all = np.ones(0)
    for feature in test_data:
        num_all+=1
        user_id_t = torch.LongTensor([feature["user_id"]]).cuda()
        movie_id_t = torch.LongTensor([feature["movie_id"]]).cuda()
        movie_cate_t = torch.LongTensor(feature["movie_cate_id_list"]).cuda()
        user_rate_t = torch.LongTensor(feature["user_rated_movie_id_list"]).cuda()
        lable_t = torch.Tensor([feature["lable"]]).cuda()
        r = model(user_id_t, movie_id_t, movie_cate_t, user_rate_t)
        test_loss+=criterion(r,lable_t)
        y_true=lable_t.cpu().numpy()
        y_true_all = np.concatenate([y_true_all, y_true])
        y_score = r.cpu()
        y_score = y_score.detach().numpy()
        y_score_all = np.concatenate([y_score_all, y_score])
    auc = sklearn.metrics.roc_auc_score(y_true_all, y_score_all)
    print('Epoch: ', epoch, '| Test_Loss: ', test_loss.item() / num_all, '| Test_AUC: ', auc)