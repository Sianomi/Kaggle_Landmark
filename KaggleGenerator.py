import keras
import csv
import os
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

LOADPATH = "D:\\ksa\\landmark-recognition-2020\\train_re"
SAVEPATH = os.getcwd()

list_subfolders_with_paths = [f.path for f in os.scandir(LOADPATH) if f.is_dir()]


with open('train.csv', mode='r') as file:
    reader = csv.reader(file)
    indexdict = {rows[0]:rows[1] for rows in reader}

# with open('train.csv', mode='r') as file:
#     reader = csv.reader(file)
#     indexdict = {}
#     i = 0
#     for rows in reader:
#         indexdict[rows[0]] = i//3
#         i = i + 1

generator = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2,
                               horizontal_flip=True, vertical_flip=False,
                               brightness_range=[0.5, 1.0], rotation_range=45,
                               zoom_range=[0.7, 1.0], fill_mode='nearest'
                               )

def generating(path, filename):
    img = load_img(os.path.join(path, filename))
    data = img_to_array(img)
    samples = expand_dims(data, 0)
    try:
        index = indexdict[filename.replace('.jpg', '')]
    except:
        print(filename)
        return
    image = generator.flow(samples, batch_size=1)
    for i in range(10):
        batch = image.next()
        savepath1 = os.path.join(os.path.join(SAVEPATH, 'train_re_arg'), str(index))
        if not os.path.isdir(savepath1):
            os.mkdir(savepath1)
        filename_split = filename.split('.')
        savepath2 = os.path.join(savepath1, filename_split[0] + '_' + str(i) + '.jpg')
        plt.imsave(fname=savepath2, arr=batch[0].astype('uint8'), format='jpeg')



for path1 in list_subfolders_with_paths:
    for path2 in os.listdir(path1):
        temp_path = os.path.join(path1, path2)
        for path3 in os.listdir(temp_path):
            temp_path2 = os.path.join(temp_path, path3)
            for filename in os.listdir(temp_path2):
                generating(temp_path2, filename)



