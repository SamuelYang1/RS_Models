Data:\
Amazon product data, Electronics, ratings only

Raw data format:\
user,item,rating,timestamp\
Example:\
AKM1MP6P0OYPR,0132793040,5.0,1365811200

Processed data format(also as input data format):\
id_user,id_item,rating\
Example:\
3017394,115856,1

Train_data:Test_data=8:2

Result:\
Epoch:  5 | Step:  2000 | Train_Loss:  0.958383573293686 | Test_Loss:  2.0478044766193415 | Test_ACC:  0.4262280520698806