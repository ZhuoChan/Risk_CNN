import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as sk
from sklearn.utils import shuffle

def read_data_atec():
    data = pd.read_csv('/data/chenzhuo/ATEC/atec_anti_fraud_train_new_0.25_half.csv')
    print (data.head(5))
    label = data['label'].values
    data.drop(['id', 'label', 'date'], axis=1, inplace=True)
    data_ = data.values  # 形成数组

    # 训练集60%, （从0,1样本集中各抽取60%）
    X_train = data_[:int(len(data_) * 0.6)]
    y_tr = label[:int(len(label) * 0.6)]

    # 验证集20%  （从0,1样本集中各抽取20%）
    X_vld = data_[int(len(data_) * 0.6):int(len(data_) * 0.8)]
    y_vld = label[int(len(label) * 0.6):int(len(label) * 0.8)]

    # 测试集20%  （从0,1样本集中各抽取20%）
    X_test = data_[int(len(data_) * 0.8):]
    y_test = label[int(len(label) * 0.8):]

    return X_train,y_tr,X_vld,y_vld,X_test,y_test

def one_hot(labels,n_class):
    labels=labels.astype(int)
    expansion=np.eye(n_class)
    y=expansion[labels]
    return y
