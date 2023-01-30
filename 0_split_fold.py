import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='/home/server/PycharmProjects/blockchain/images/Images/')
parser.add_argument("--fold", type=int, default=5)
args = parser.parse_args()

os.makedirs('./dataset', exist_ok=True)
data_path = args.data_path
new_path = './dataset/'
fold = args.fold

os.makedirs(new_path, exist_ok=True)

for i in range(fold):
    os.makedirs(new_path + '/' + str(i+1), exist_ok=True)

folder_path = os.listdir(data_path)
folder_path.sort()

for i in folder_path:
    file_path = os.listdir(data_path + '/' + i)
    file_path.sort()
    split_fold = len(file_path) // fold
    for j in range(fold):
        if j == fold-1:
            split_list = file_path[j*split_fold:]
        else:
            split_list = file_path[j*split_fold: j*split_fold+split_fold]
            
        for k in split_list :
            os.makedirs(new_path + '/' + str(j+1) + '/' + i + '/', exist_ok=True)
            shutil.copy(data_path + '/' + i + '/' + k, new_path + '/' + str(j+1) + '/' + i + '/' + k )

            print(data_path + '/' + i + '/' + k)
            print('\t COPY -->', new_path + '/' + str(j+1) + '/' + i + '/' + k )



