from tensorflow.keras import applications
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pandas as pd

img_width, img_height = 224, 224
model = applications.InceptionResNetV2(include_top=False, weights='imagenet',pooling='avg') # pretrained network without fully-connected layer

with open("myn_bottoms.txt", "w") as a:
    for root, dirs, files in os.walk("./Data/Myntra/Bottom", topdown=False):
        for name in files:
            f = os.path.join(root, name)
            a.write(str(f) + os.linesep)

tops_path = pd.read_csv("myn_bottoms.txt",header=None, names=['Path'])
tops_features = []
tops_title = []
error_value = []
print(len(tops_path["Path"]))
j = 0
for i in tops_path['Path']:
    try:
        img = image.load_img(i, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis=0)
        x = preprocess_input(x)
        extracted_features = model.predict(x)
        tops_features.append(extracted_features.squeeze())
        image_title = i.split("/")[-1]
        tops_title.append(image_title)
        j+=1
        if j%1000==0:
            print("Current index:",j)
    except:
        error_value.append(i)

tops_features = np.array(tops_features)
print(tops_features.shape)
tops_title = np.array(tops_title)
tops_errors = np.array(error_value)
tops_features = tops_features.reshape((tops_features.shape[0],1536)) # after pooling dimensions of inception-resnet
print(tops_features.shape)
print(tops_title.shape)

np.save(open('./similarity_features/Myntra_Bottoms_Incep_features.npy', 'wb'), tops_features)
np.save(open('./similarity_features/Myntra_Bottoms_Incep_feature_img_ids.npy', 'wb'), tops_title)
np.save(open('./similarity_features/Myntra_Bottoms_Incep_Errors.npy', 'wb'), tops_errors)
