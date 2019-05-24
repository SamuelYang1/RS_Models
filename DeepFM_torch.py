import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
class DeepFM(nn.Module):
    def __init__(self,fieldsize=3,feasize=100,k=5):
        super(DeepFM, self).__init__()
        # build FM
        self.w=nn.Embedding(feasize,1)
        self.v=nn.Embedding(feasize,k)
        self.w0=nn.Parameter(torch.zeros(1))

        #build DNN
        hidden_layer = [10, 5,3]
        self.dnn=torch.nn.Sequential()
        self.dnn.add_module("hidden_0",nn.Linear(fieldsize*k,hidden_layer[0]))
        self.dnn.add_module("Relu_0",nn.ReLU())
        for i in range(1,len(hidden_layer)):
            self.dnn.add_module("hidden_"+str(i),nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            self.dnn.add_module("Relu_"+str(i),nn.ReLU())
        self.dnn.add_module("Out",nn.Linear(hidden_layer[len(hidden_layer)-1],1))



    def embedding(self,input):
        return input
    def forward(self, feature):
        res = torch.zeros(1)
        #2-order
        V=self.v(torch.LongTensor(feature))
        d=list(V.shape)[0]
        for i in range(d):
            for j in range(i+1,d):
                res+=V[i]@(V[j].reshape(-1,1))
        #1-order
        W=self.w(torch.LongTensor(feature))
        res+=torch.sum(W)
        #0-order
        res+=self.w0
        #dnn
        res+=self.dnn(V.reshape(-1))
        #return(torch.sigmoid(res))
        return res

model=DeepFM()
#print(model([6,7,8]))
# params=list(model.parameters())
# print(params)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = F.binary_cross_entropy_with_logits
for i in range(4000):
    x=torch.tensor([6,7,8])
    y_=model(x)
    y=torch.tensor([0.0])
    loss = criterion(y_, y)
    optimizer.zero_grad()
    print(loss)
    loss.backward()
    optimizer.step()