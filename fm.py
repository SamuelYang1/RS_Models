import pandas as pd
from itertools import count
from collections import defaultdict
import numpy as np
from scipy.sparse import csr
import tensorflow as tf
def one_hot(dic,ind=None,p=None):
    if ind==None:
        i=count(0)
        ind=defaultdict(lambda :next(i))
    n=len(list(dic.values())[0])
    g=len(list(dic.keys()))
    nz=n*g
    s=0
    col=np.empty(nz)
    for key,values in dic.items():
        col[s::g]=[ind[key+str(val)]for val in values]
        s+=1
    row=np.repeat(np.arange(0,n),g)
    data = np.ones(nz)
    if p==None:
        p=len(ind)
    legal = np.where(col < p)
    return csr.csr_matrix((data[legal], (row[legal], col[legal])),shape=(n,p)), ind

cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('data/ua.base', delimiter='\t', names=cols)
test = pd.read_csv('data/ua.test', delimiter='\t', names=cols)

X_train, ind = one_hot({'users': train['user'].values, 'items': train['item'].values})
X_test, ind = one_hot({'users': test['user'].values, 'items': test['item'].values}, ind,X_train.shape[1])
Y_train = train['rating'].values
Y_test= test['rating'].values
X_train = X_train.todense()
X_test = X_test.todense()

print(X_train.shape)
print(X_test.shape)

_,p=X_train.shape
k=10
x=tf.placeholder("float",shape=[None,p])
y=tf.placeholder("float",shape=[None,1])
w0=tf.Variable(tf.zeros([1]))
w=tf.Variable(tf.zeros([p,1]))
v=tf.Variable(tf.zeros([p,k]))
inter=0.5*tf.reduce_sum(tf.pow(tf.matmul(x,v),2)-tf.matmul(tf.pow(x,2),tf.pow(v,2)),1,keepdims=True)
y_=w0+inter+tf.matmul(x,w)
loss=tf.reduce_mean(tf.square(y-y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]
    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))
    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)

epochs = 10
batch_size = 50
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for epoch in range(epochs):
    perm = np.random.permutation(X_train.shape[0])
    for bX, bY in batcher(X_train[perm], Y_train[perm], batch_size):
        _,res=sess.run([optimizer,loss],feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)})
        print(res)
errors = []
for bX, bY in batcher(X_test, Y_test):
    errors.append(sess.run(loss, feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)}))
RMSE = np.sqrt(np.array(errors).mean())
print('RMSE=',RMSE)
sess.close()