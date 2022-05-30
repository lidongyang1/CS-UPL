# -*- coding: utf-8 -*-
# @Time    : 2022/2/21 16:17
# @Author  : cmdxmm
# @FileName: getDifferentUserData_less_30000.py
# @Email   ：lidongyang@mail.sdu.edu.cn
import numpy as np
from tensorflow.keras import  Model
from stellargraph.layer import GCN_LSTM
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
import tensorflow.keras as K
import heapq
from tensorflow.keras.utils import plot_model


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

####将数据分为训练集和测试集
def train_test_split(data, train_portion):
    time_len = data.shape[1]
    train_size = int(time_len * train_portion)
    train_data = np.array(data.iloc[:, :train_size])
    test_data = np.array(data.iloc[:, train_size:])
    return train_data, test_data

###对数据进行归一化
def scale_data(train_data, test_data):
    # print(np.argwhere(np.isnan(train_data)))
    # print(np.argwhere(np.isnan(test_data)))
    train_data = np.abs(train_data)
    test_data = np.abs(test_data)
    max_speed = train_data.max()
    min_speed = train_data.min()
    print('max：',max_speed)
    print('min：',min_speed)
    train_scaled = (train_data - min_speed) / (max_speed - min_speed)
    test_scaled = (test_data - min_speed) / (max_speed - min_speed)
    return train_scaled, test_scaled

#将数据转成10个step训练1个step的格式
def sequence_data_preparation(seq_len, pre_len, train_data, test_data):
    trainX, trainY, testX, testY = [], [], [], []

    for i in range(train_data.shape[1] - int(seq_len + pre_len - 1)):
        a = train_data[:, i : i + seq_len + pre_len]
        trainX.append(a[:, :seq_len])
        trainY.append(a[:, -1])

    for i in range(test_data.shape[1] - int(seq_len + pre_len - 1)):
        b = test_data[:, i : i + seq_len + pre_len]
        testX.append(b[:, :seq_len])
        testY.append(b[:, -1])

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)

    return trainX, trainY, testX, testY

#将数据转成10个step训练1个step的格式
def sequence_data_preparationOtherUsers(seq_len, pre_len, train_data, test_data):
    trainX, trainY, testX, testY = [], [], [], []

    for i in range(train_data.shape[1] - int(seq_len + pre_len - 1)):
        a = train_data[:, i : i + seq_len + pre_len]
        trainX.append(a[:, -2])
        trainY.append(a[:, -1])

    for i in range(test_data.shape[1] - int(seq_len + pre_len - 1)):
        b = test_data[:, i : i + seq_len + pre_len]
        testX.append(b[:, -2])
        testY.append(b[:, -1])

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)

    return trainX, trainY, testX, testY

def  attention_block(inputs,time_stpes):
    input_dim =  int(inputs.shape[2]) # shape = (batch_size, time_steps, input_dim)

    a = keras.layers.Permute((2, 1))(inputs) # shape = (batch_size, input_dim, time_steps)

    a = keras.layers.Reshape((input_dim, time_stpes))(a) # this line is not useful. It's just to know which dimension is what.

    a = tf.keras.layers.Dense(time_stpes, activation='softmax')(a)# 为了让输出的维数和时间序列数相同（这样才能生成各个时间点的注意力值）

    a = keras.layers.Lambda(lambda x: K.backend.mean(x, axis=1), name='dim_reduction')(a)
    a = keras.layers.RepeatVector(input_dim)(a)

    a_probs = keras.layers.Permute((2, 1), name='attention_vec')(a) # shape = (batch_size, time_steps, input_dim)

    output_attention_mul = keras.layers.Multiply()([inputs, a_probs]) #把注意力值和输入按位相乘，权重乘以输入

    return output_attention_mul

userId = 0 #当前为用户1的相似用户辅助信息
top_k = 3
seq_len = 24
pre_len = 25

view_counts = np.load('./YouTube_NewDATA/data/userData/users=30_FileNum=400/User_1_Contents.npy')
video_dist_adj = np.load('./YouTube_NewDATA/data/userData/users=30_FileNum=400/Content_EdgeMatrix.npy')
view_counts= pd.DataFrame(view_counts)
num_nodes, time_len = view_counts.shape
train_rate = 0.8
print(view_counts.shape)
train_data, test_data = train_test_split(view_counts, train_rate)
print("Train data: ", train_data.shape)
print("Test data: ", test_data.shape)
trainX, trainY, testX, testY = sequence_data_preparation(
    seq_len, pre_len, train_data, test_data
)


SocialRelationship = np.load('./YouTube_NewDATA/data/userData/users=30_FileNum=400/socialRelationship.npy')
SocialSimilarityUsers = np.delete(heapq.nlargest(top_k+1,range(len(SocialRelationship[userId])),SocialRelationship[userId].take), userId, axis=0)
print(SocialSimilarityUsers[0])
view_counts_top_1 = np.load('./YouTube_NewDATA/data/userData/users=30_FileNum=400/User_'+str(SocialSimilarityUsers[0]+1)+'_Contents.npy')
view_counts_top_2 = np.load('./YouTube_NewDATA/data/userData/users=30_FileNum=400/User_'+str(SocialSimilarityUsers[1]+1)+'_Contents.npy')
view_counts_top_3 = np.load('./YouTube_NewDATA/data/userData/users=30_FileNum=400/User_'+str(SocialSimilarityUsers[2]+1)+'_Contents.npy')

view_counts_top_1 = pd.DataFrame(view_counts_top_1)
view_counts_top_2 = pd.DataFrame(view_counts_top_2)
view_counts_top_3 = pd.DataFrame(view_counts_top_3)

train_data_top_1, test_data_top_1 = train_test_split(view_counts_top_1, train_rate)
train_data_top_2, test_data_top_2 = train_test_split(view_counts_top_2, train_rate)
train_data_top_3, test_data_top_3 = train_test_split(view_counts_top_3, train_rate)


print("Train data: ", train_data_top_1.shape)
print("Test data: ", test_data_top_1.shape)
trainX_top_1, _, testX_top_1, _ = sequence_data_preparationOtherUsers(
    seq_len, pre_len, train_data_top_1, test_data_top_1
)
trainX_top_2, _, testX_top_2, _ = sequence_data_preparationOtherUsers(
    seq_len, pre_len, train_data_top_2, test_data_top_2
)
trainX_top_3, _, testX_top_3, _ = sequence_data_preparationOtherUsers(
    seq_len, pre_len, train_data_top_3, test_data_top_3
)

gcn_lstm = GCN_LSTM(
    seq_len=seq_len,
    adj=video_dist_adj,
    gc_layer_sizes=[32, 64],
    gc_activations=["relu", "relu"],
    lstm_layer_sizes=[32, 64],
    lstm_activations=["relu","relu"],
    dropout=0.2,
)
x_input, x_output1 = gcn_lstm.in_out_tensors()
x_output1 = keras.layers.Dense(200,activation='relu')(x_output1)

input1 = tf.keras.Input(shape=(400,24), name='input1')
x1 = keras.layers.Bidirectional(keras.layers.LSTM(32,activation='tanh',return_sequences=True))(input1)
x2 = keras.layers.Bidirectional(keras.layers.LSTM(64,activation='tanh',return_sequences=False))(x1)

x3 = tf.keras.layers.Dense(512,activation='relu')(x2)
x_output = keras.layers.Dense(400,activation='relu')(x3)


input_top_1 = tf.keras.Input(shape=(400), name='input_top_1')
input_top_2 = tf.keras.Input(shape=(400), name='input_top_2')
input_top_3 = tf.keras.Input(shape=(400), name='input_top_3')

x1_input_top_1 = keras.layers.Dense(512,activation='relu')(input_top_1)
x2_input_top_1 = keras.layers.Dense(100,activation='relu')(x1_input_top_1)

x1_input_top_2 = keras.layers.Dense(512,activation='relu')(input_top_2)
x2_input_top_2 = keras.layers.Dense(100,activation='relu')(x1_input_top_2)


x1_input_top_3 = keras.layers.Dense(512,activation='relu')(input_top_3)
x2_input_top_3 = keras.layers.Dense(100,activation='relu')(x1_input_top_3)

top_k_output = keras.layers.Concatenate()([x2_input_top_1,x2_input_top_2,x2_input_top_3])

top_k_output = keras.layers.Reshape((3,100))(top_k_output)

top_k_output =  attention_block(top_k_output,3)
#

top_k_output = keras.layers.Flatten()(top_k_output)
#
# top_k_output = keras.layers.Dense(100)(top_k_output)

# ,top_k_output
x_output_f1 = keras.layers.Concatenate()([x_output1,x_output])
x_output_f1 = keras.layers.Dense(100,activation='relu')(x_output_f1)

x_output_f2 = keras.layers.Concatenate()([x_output,top_k_output])
x_output_f2 = keras.layers.Dense(100,activation='relu')(x_output_f2)


x_output_f = keras.layers.Concatenate()([x_output,x_output_f1,x_output_f2])
x_output_f = keras.layers.Dense(512,activation='relu')(x_output_f)
x_output_f = keras.layers.Dense(400)(x_output_f)

model = Model(inputs=[x_input,input1,input_top_1,input_top_2,input_top_3], outputs=x_output_f)

adam = tf.keras.optimizers.Adam(learning_rate=0.001)
history = model.compile(optimizer=adam, loss="mae", metrics=["mae"])
checkpoint = ModelCheckpoint(filepath='model/CS-GCN-LSTM.h5', monitor='val_loss', mode='auto', save_best_only='True', verbose=1)
history = LossHistory()
callback_lists = [checkpoint,history]
model.summary()
plot_model(model,to_file='model.png',show_shapes=True)

history2 =  model.fit(
    [trainX,trainX,trainX_top_1,trainX_top_2,trainX_top_3],
    trainY,
    epochs=100,
    batch_size=8,
    shuffle=True,
    verbose=1,
    validation_data=[[testX,testX,testX_top_1,testX_top_2,testX_top_3], testY],
    callbacks=callback_lists
)

print(
    "Train loss: ",
    history2.history["loss"][-1],
    "\nTest loss:",
    history2.history["val_loss"][-1],
)
losses = history.losses
print(losses)
np.save('./model/losses/'
        'losses.npy',losses)
