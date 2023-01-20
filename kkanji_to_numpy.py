from PIL import Image
import os
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder

ROOT_DIR = "./kkanji2"

dir_list = os.listdir(ROOT_DIR)
# print(dir_list)

y_all = []
x_all = []
for dir_name in dir_list:
    file_list = os.listdir(ROOT_DIR + "/" + str(dir_name))
    for file_name in file_list:
        image = Image.open(ROOT_DIR + "/" + str(dir_name) + "/" + file_name)
        # convert image to numpy array
        data = np.asarray(image)
        y_all.append(dir_name)
        x_all.append(data)

le = LabelEncoder()
y_all = le.fit_transform(y_all)

random.seed(0)
random.shuffle(x_all)
random.seed(0)
random.shuffle(y_all)

# Put aside a few samples to create our test set
TEST_SAMPLES = 10000
x_test, y_test = x_all[:TEST_SAMPLES], y_all[:TEST_SAMPLES]
x_train, y_train = x_all[TEST_SAMPLES:], y_all[TEST_SAMPLES:]

np.savez("kkanji-train-imgs.npz", x_train)
np.savez("kkanji-train-labels.npz", y_train)
np.savez("kkanji-test-imgs.npz", x_test)
np.savez("kkanji-test-labels.npz", y_test)
