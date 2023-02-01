import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='./Images/')
parser.add_argument("--annotation_path", type=str, default='./Annotation/')
args = parser.parse_args()

data_path = args.data_path
annotation_path = args.annotation_path

new_path = './extract_images/'
os.makedirs(new_path, exist_ok=True)


from PIL import Image
import re

def extract_image(ano_path, img_path ,new_path, name):
    f = open(ano_path, 'r')
    im = Image.open(img_path)
    crop_list = []
    count = 0
    for i in f:
        if 'xmin' in i:
            xmin = re.sub(r'[^0-9]', '', i)
            crop_list.append(int(xmin))
        if 'ymin' in i:
            ymin = re.sub(r'[^0-9]', '', i)
            crop_list.append(int(ymin))
        if 'xmax' in i:
            xmax = re.sub(r'[^0-9]', '', i)
            crop_list.append(int(xmax))
        if 'ymax' in i:
            ymax = re.sub(r'[^0-9]', '', i)
            crop_list.append(int(ymax))
        if len(crop_list) == 4:
            print(count, crop_list)
            extract_img = im.crop(crop_list)
            try:
                extract_img = extract_img.save(new_path + name + '_' + str(count) + '.jpg')
            except:
                extract_img = extract_img.convert('RGB')
                extract_img = extract_img.save(new_path + name + '_' + str(count) + '.jpg')

            crop_list = []
            count =+ 1


for i in os.listdir(annotation_path):
    fold_path = annotation_path + '/' + i
    file_name_list = os.listdir(fold_path)
    for j in file_name_list:
        ano_path = annotation_path + '/' + i + '/' + j
        img_path = data_path + '/' + i + '/' + j + '.jpg'
        new_final_path = new_path + '/' + i + '/'
        os.makedirs(new_final_path, exist_ok=True)
        extract_image(ano_path, img_path, new_final_path, j)






