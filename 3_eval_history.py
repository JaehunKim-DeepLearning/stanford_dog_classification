import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--histroy_path", type=str, default='./history/')
parser.add_argument("--weights", type=str, default='imagenet') # noisystudent advprob autoaugment imagenet
parser.add_argument("--affix", type=str, default='')
parser.add_argument("--mode", type=str, default='B0')
parser.add_argument("--fold", type=int, default=5)
args = parser.parse_args()

fold = args.fold
weights = args.weights
mode = args.mode
affix = args.affix
histroy_path = args.histroy_path

acc_list = [] 
val_acc_list = []

print("\n" + weights.upper() + ' ' + mode + " MODEL")

print("\nFOLD HISTORY RESULT OF BEST MODEL")

plt.subplots_adjust(hspace=1)
for i in range(1, fold+1):
    if affix != '':
        df = pd.read_csv(histroy_path + '/' + affix + '_' + weights + '_' + mode + '_FOLD' + str(i) + '_history.csv')
    else:
        df = pd.read_csv(histroy_path + '/' + weights + '_' + mode + '_FOLD' + str(i) + '_history.csv')

    plt.subplot(fold, 2, i*2-1)
    plt.plot(df['accuracy'])
    plt.plot(df['val_accuracy'])
    plt.title('%s Model FOLD%d accuracy' %(mode, i))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Valid'], loc='upper left', fontsize = 8)

    plt.subplot(fold, 2, i*2)
    plt.plot(df['loss'])
    plt.plot(df['val_loss'])
    plt.title('%s Model FOLD%d loss' % (mode, i))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.legend(['Train', 'Valid'], loc='upper left')

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
plt.show()

