from keras import applications
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pandas as pd
img_width, img_height = 224, 224
train_data_dir = "/home/ubuntu/capstone/Data/Flipkart/"
model = applications.ResNet50(include_top=False, weights='imagenet')

with open("flip_tops.txt", "w") as a:
    for root, dirs, files in os.walk("/home/ubuntu/capstone/Data/Flipkart/Tops", topdown=False):
        for name in files:
            f = os.path.join(root, name)
            a.write(str(f) + os.linesep)

tops_path = pd.read_csv("/home/ubuntu/capstone/flip_tops.txt",header=None, names=['Path'])
tops_features = []
tops_title = []
error_value = []

for i in tops_path['Path']:
    try:
        img = image.load_img(i, target_size=(224,224))
        x = image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        extracted_features = model.predict(x)
        tops_features.append(extracted_features.squeeze())
        image_title = i.split("/")[-1]
        tops_title.append(image_title)
    except:
        print(i)
        error_value.append(i)

tops_features = np.array(tops_features)
tops_title = np.array(tops_title)
tops_errors = np.array(error_value)
tops_features = tops_features.reshape((tops_features.shape[0],100352))
print(tops_features.shape)
print(tops_title.shape)

np.save(open('./Flipkart_Tops_ResNet_features.npy', 'wb'), tops_features)
np.save(open('./Flipkart_Tops_ResNet_feature_img_ids.npy', 'wb'), tops_title)
np.save(open('./Flipkart_Tops_ResNet_Errors.npy', 'wb'), tops_errors)



