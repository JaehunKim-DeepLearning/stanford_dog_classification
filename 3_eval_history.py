import os
import shutil
import pandas as pd
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--histroy_path", type=str, default='./history/new')
parser.add_argument("--weights", type=str, default='noisystudent') # noisystudent advprob autoaugment imagenet
parser.add_argument("--mode", type=str, default='B5')
parser.add_argument("--fold", type=int, default=5)
args = parser.parse_args()

fold = args.fold
weights = args.weights
mode = args.mode
histroy_path = args.histroy_path

acc_list = [] 
val_acc_list = []

print("\n" + weights.upper() + ' ' + mode + " MODEL")

print("\nFOLD HISTORY RESULT OF BEST MODEL")

for i in range(1, fold+1):
    df = pd.read_csv(histroy_path + '/' + weights + '_' + mode + '_FOLD' + str(i)  +'_history.csv')
    df = df.sort_values(by=['val_accuracy'], ascending=False)
    df = df.iloc[0]

    acc = df['accuracy']
    val_acc = df['val_accuracy']
    loss = df['loss']
    val_loss = df['val_loss']

    result_str = "\tFOLD%d \tloss : %0.4f \tacc : %0.4f \tval_loss : %0.3f \tval_acc : %0.3f" %(i, loss, acc, val_loss, val_acc)
    print(result_str)

    acc_list.append(acc)
    val_acc_list.append(val_acc)

print("\nRESULT AVERAGE")
print("\tTRAIN : %0.3f" %(sum(acc_list)/len(acc_list)))
print("\tVALID : %0.3f" %(sum(val_acc_list)/len(val_acc_list)))
print()
