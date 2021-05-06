#Load the necessary files
import pandas as pd
import numpy as np
import json
import ast
import tensorflow as tf
from keras import applications
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image

#Load the model
model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet',pooling='avg')

#Load the darknet results on all flipkart images - This is obtained by running test command on darknet folder
flipkart = pd.read_csv("./flipkart_results.csv")
Lower = '|'.join(['Bottom', 'Shorts', 'Pants'])
Upper = '|'.join(['Hoodie', 'Jackets', 'Tops'])
Full = '|'.join(['Jumpsuit', 'Cloak', 'Dress'])
flipkart.loc[flipkart['category'].str.contains(Lower), "actual_cat"] = "Lower"
flipkart.loc[flipkart['category'].str.contains(Upper), "actual_cat"] = "Upper"
flipkart.loc[flipkart['category'].str.contains(Full), "actual_cat"] = "Full"
#Filter one category at time
filename = list(flipkart[flipkart['actual_cat'] == 'Upper']['filename'])
objects = list(flipkart[flipkart['actual_cat'] == 'Upper']['objects'])
pred_cat = list(flipkart[flipkart['actual_cat'] == 'Upper']['pred_cat'])
actual_cat = list(flipkart[flipkart['actual_cat'] == 'Upper']['actual_cat'])
upper_features = []
upper_title = []
lower_features = []
lower_title = []
full_features = []
full_title = []
error_value = []
# Below loop will go into each image and crop the predicted category from it
# After evaluating performance of the object detection, we could set logic to go over the images whose detection threshold >= 0.70
for file, objec, pre_cat, act_cat in zip(filename, objects, pred_cat, actual_cat):
    try:
        ls = pre_cat.strip('[]').replace("'", "").split(', ')
        objec = ast.literal_eval(objec.strip('[]'))
        if len(ls) >= 1:
            for clas in objec:
                if clas['confidence'] >= 0.70:
                    predict_image = Image.open("/home/ubuntu/capstone/" + file)
                    center_x = clas['relative_coordinates']['center_x']
                    center_y = clas['relative_coordinates']['center_y']
                    wid = clas['relative_coordinates']['width']
                    heig = clas['relative_coordinates']['height']
                    width = wid / 2.0
                    height = heig / 2.0
                    img_wid = predict_image.size[0]
                    img_ht = predict_image.size[1]
                    left = (center_x - width) * img_wid
                    right = (center_x + width) * img_wid
                    bottom = (center_y + height) * img_ht
                    top = (center_y - height) * img_ht
                    predict_image = predict_image.crop((left, top, right, bottom))
                    predict_image.save("croped_image.jpg")
                    predict_image = Image.open('croped_image.jpg')
                    img = image.load_img('croped_image.jpg', target_size=(224, 224))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    features = model.predict(x)
                    if clas['name'] == 'Upper':
                        upper_features.append(features.squeeze())
                        image_title = file.split("/")[-1]
                        upper_title.append(image_title)
                    elif clas['name'] == 'Lower':
                        lower_features.append(features.squeeze())
                        image_title = file.split("/")[-1]
                        lower_title.append(image_title)
                    elif clas['name'] == 'Full':
                        full_features.append(features.squeeze())
                        image_title = file.split("/")[-1]
                        full_title.append(image_title)
    except:
        i = file.split("/")[-1]
        print(i)
        error_value.append(i)

upper_features = np.array(upper_features)
print(upper_features.shape)
lower_features = np.array(lower_features)
print(lower_features.shape)
full_features = np.array(full_features)
print(full_features.shape)

upper_title = np.array(upper_title)
lower_title = np.array(lower_title)
full_title = np.array(full_title)
upper_errors = np.array(error_value)
upper_features = upper_features.reshape(
    (upper_features.shape[0], upper_features.shape[1]))
lower_features = lower_features.reshape(
    (lower_features.shape[0], lower_features.shape[1]))
full_features = full_features.reshape(
    (full_features.shape[0], full_features.shape[1]))
print(upper_features.shape)
print(lower_features.shape)
print(full_features.shape)


np.save(open('./Flipkart_Upper_InceptionResnet_features_avg.npy', 'wb'), upper_features)
np.save(open('./Flipkart_Upper_InceptionResnet_feature_img_ids_avg.npy', 'wb'), upper_title)

np.save(open('./Flipkart_Lower_InceptionResnet_features_avg.npy', 'wb'), lower_features)
np.save(open('./Flipkart_Lower_InceptionResnet_feature_img_ids_avg.npy', 'wb'), lower_title)

np.save(open('./Flipkart_Full_InceptionResnet_features_avg.npy', 'wb'), full_features)
np.save(open('./Flipkart_Full_InceptionResnet_feature_img_ids_avg.npy', 'wb'), full_title)
np.save(open('./Flipkart_InceptionResnet_Errors_avg.npy', 'wb'), upper_errors)
