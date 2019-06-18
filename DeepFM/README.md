$ python3 DeepFM.py

Data:\
Criteo Dataset(Kaggle Display Advertising Challenge Dataset)\
http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

Raw data format:\
Label I1-I13 C1-C26\
Example:\
0	1	1	5	0	1382	4	15	2	181	1	2268fd1e64	80e26c9b	fb936136	7b4723c4	25c83c98	7e0ccccf	de7995b8	1f89b562	a73ee510	a8cd5504	b2cb9c98	37c9c164	2824a5f6	1adce6ef	8ba8b39a	891b62e7	e5ba7672	f54016b9	21ddcdc9	b1252a9d	07b5194c		3a171ecb	c5c50484	e8b83407	9727dd16

Processed data format(also as input data format):\
Label,Id_I1-Id_I13,Id_C1-Id_C26\
Example:\
0,12811,195347,1159195,391633,969195,1547192,1875506,745669,1329392,226875,1445895,1223315,337610,643101,754484,211145,1554641,935321,722328,117629,1211251,584776,1818086,1406663,319270,1977696,1480572,1178521,209779,1557301,1695603,419761,2031398,1228261,2054098,1639104,1885158,1489478,2033774

Train_data:Test_data=9:1

Result:\
Epoch:  12 | Step:  120000 | Train_Loss:  0.6039575296282769\
Epoch:  12 | Test_Loss:  0.6392237342153351 | Test_ACC:  0.7584234379027556 | Test_AUC:  0.7409754304542977