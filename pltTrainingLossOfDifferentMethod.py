# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 7:54
# @Author  : cmdxmm
# @FileName: plotPredictAndGroundTrue_Matrix.py
# @Email   ：lidongyang@mail.sdu.edu.cn

import numpy as np
import matplotlib.pyplot as plt
oneUser_CS_GCN_LSTM = np.load('./model/losses/losses.npy')
oneUser_S_GCN_LSTM = np.load('./model/losses/S-GCN-LSTM_losses.npy')
oneUser_C_GCN_LSTM = np.load('./model/losses/C-GCN-LSTM_losses.npy')
oneUser_GCN_LSTM = np.load('./model/losses/GCN-LSTM_losses.npy')
AllUser_CS_GCN_LSTM = np.load('./model/losses/losses_allUsers.npy')
AllUser_S_GCN_LSTM = np.load('./model/losses/S-GCN-LSTM_losses_allUsers.npy')
AllUser_C_GCN_LSTM = np.load('./model/losses/C-GCN-LSTM_losses_allUsers.npy')
AllUser_GCN_LSTM = np.load('./model/losses/GCN-LSTM_losses_allUsers.npy')
labels=['UPL(Single User)','C-UPL(Single User)','S-UPL(Single User)','CS-UPL(Single User)','UPL(Multiple Users)','C-UPL(Multiple Users)','S-UPL(Multiple Users)','CS-UPL(Multiple Users)']
x = np.arange(100)

# plt.figure(figsize=(8, 5))
plt.figure()
plt.plot(x,oneUser_GCN_LSTM,label=labels[0],linewidth=2)
plt.plot(x,oneUser_C_GCN_LSTM,label=labels[1],linewidth=2)
plt.plot(x,oneUser_S_GCN_LSTM,label=labels[2],linewidth=2)
plt.plot(x,oneUser_CS_GCN_LSTM,label=labels[3],linewidth=2)
plt.legend(ncol=2,fontsize=11.5)
plt.xlabel('Epochs', fontsize=15,fontweight='bold')
plt.ylabel('Training Loss', fontsize=15,fontweight='bold')
plt.grid(ls='--')
plt.show()



plt.figure()
plt.plot(x,AllUser_GCN_LSTM,label=labels[4],linewidth=2)
plt.plot(x,AllUser_C_GCN_LSTM,label=labels[5],linewidth=2)
plt.plot(x,AllUser_S_GCN_LSTM,label=labels[6],linewidth=2)
plt.plot(x,AllUser_CS_GCN_LSTM,label=labels[7],linewidth=2)
plt.legend(ncol=2,fontsize=10.5)
plt.xlabel('Epochs', fontsize=15,fontweight='bold')
plt.ylabel('Training Loss', fontsize=15,fontweight='bold')
plt.grid(ls='--')
plt.show()

plt.figure()
plt.plot(x,oneUser_GCN_LSTM,label=labels[0],linewidth=2)
plt.plot(x,oneUser_C_GCN_LSTM,label=labels[1],linewidth=2)
plt.plot(x,oneUser_S_GCN_LSTM,label=labels[2],linewidth=2)
plt.plot(x,oneUser_CS_GCN_LSTM,label=labels[3],linewidth=2)
plt.plot(x,AllUser_GCN_LSTM,label=labels[4],linewidth=2)
plt.plot(x,AllUser_C_GCN_LSTM,label=labels[5],linewidth=2)
plt.plot(x,AllUser_S_GCN_LSTM,label=labels[6],linewidth=2)
plt.plot(x,AllUser_CS_GCN_LSTM,label=labels[7],linewidth=2)
plt.legend(ncol=2,fontsize=10.5)
plt.xlabel('Epochs', fontsize=15,fontweight='bold')
plt.ylabel('Training Loss', fontsize=15,fontweight='bold')
plt.grid(ls='--')
plt.show()