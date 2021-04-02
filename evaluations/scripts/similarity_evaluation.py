import scipy.spatial
import tensorflow as tf
from keras import applications
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt


extracted_features = np.load('./similarity_features/Flipkart_Upper_InceptionResnet_features.npy')
img_ids = np.load('./similarity_features/Flipkart_Upper_InceptionResnet_feature_img_ids.npy')
print(extracted_features.shape)

upper_ex_fea = np.load('./Flipkart_Upper_InceptionResnet_features_avg.npy')
img_ids_upp = np.load('./Flipkart_Upper_InceptionResnet_feature_img_ids_avg.npy')
print(upper_ex_fea.shape)
lower_ex_fea = np.load('./Flipkart_Lower_InceptionResnet_features_avg.npy')
img_ids_low = np.load('./Flipkart_Lower_InceptionResnet_feature_img_ids_avg.npy')
full_ex_fea = np.load('./Flipkart_Full_InceptionResnet_features_avg.npy')
img_ids_full = np.load('./Flipkart_Full_InceptionResnet_feature_img_ids_avg.npy')

path = "/home/ubuntu/capstone/Images_test/white.jpg"
command_darknet = "./darknet detector test data/obj.data cfg/yolo-obj.cfg yolo-obj_final.weights -dont_show -thresh 0.3 -out result.json < test_sample.txt"
os.chdir("./darknet")
file_newpath = "echo "+path+" > test_sample.txt"
os.system(file_newpath)
os.system(command_darknet)
results = pd.read_json("./result.json")
predict_image = Image.open(path)
for i in results['objects']:
    if len(i) > 1:
        views = 1
        for clas in i:
            if clas['confidence'] <= 0.40:
                views = 0
        if views != 0:
            predict_image = Image.open('predictions.jpg')
        print("More than one objects found")
    elif len(i) == 1:
        views = 1
        for clas in i:
            if clas['confidence'] >= 0.70:
                predict_image = Image.open(path)
                plt.figure(figsize=(20, 10))
                plt.subplot(3, 5, 3)
                img = image.load_img(path)
                x = image.img_to_array(img)
                plt.imshow(x.astype('uint8'))
                plt.title("Inception ResNet \n\n Example 15 \n\n Original Image")
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
                model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')
                img = image.load_img('croped_image.jpg', target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                query_features = model.predict(x)
                query_features = np.array(query_features).reshape((1,query_features.shape[1] ))
                print(clas['name'])
                print("Cropped Version")
                print(query_features)
                if clas['name'] == 'Upper':
                    distances = scipy.spatial.distance.cdist(upper_ex_fea, query_features, "cosine")
                    print(distances.shape)
                    results = zip(range(len(distances)), distances)
                    results = sorted(results, key=lambda x: x[1])
                    print(results[0:5])
                    img_id = []
                    scores = []
                    for id, score in results[0:5]:
                        print(img_ids_upp[id])
                        print(score)
                        img_id.append(img_ids_upp[id])
                        scores.append((1-score)*100)
                    columns = 5
                    for i, images in enumerate(img_id):
                        plt.subplot(3,5,i+6)
                        if os.path.isfile("/home/ubuntu/capstone/Data/Flipkart/Tops/"+images):
                            img = image.load_img("/home/ubuntu/capstone/Data/Flipkart/Tops/"+images)
                        elif os.path.isfile("/home/ubuntu/capstone/Data/Flipkart/Hoodie/"+images):
                            img = image.load_img("/home/ubuntu/capstone/Data/Flipkart/Hoodie/" + images)
                        elif os.path.isfile("/home/ubuntu/capstone/Data/Flipkart/Jackets/"+images):
                            img = image.load_img("/home/ubuntu/capstone/Data/Flipkart/Jackets/" + images)
                        x = image.img_to_array(img)
                        plt.imshow(x.astype('uint8'))
                        if ((i+6) == 8):
                            plt.title("Cropped Version\n\n"+img_id[i]+" _ "+str(np.round(scores[i],2))+"%")
                        else:
                            plt.title(img_id[i]+" _ "+str(np.round(scores[i],2))+"%")
                elif clas['name'] == 'Lower':
                    distances = scipy.spatial.distance.cdist(lower_ex_fea, query_features, "cosine")
                    print(distances.shape)
                    results = zip(range(len(distances)), distances)
                    results = sorted(results, key=lambda x: x[1])
                    results.to_csv("results_purple.csv", index=False)
                    print(results[0:5])
                    img_id = []
                    scores = []
                    for id, score in results[0:5]:
                        print(img_ids_low[id])
                        print(score)
                        img_id.append(img_ids_low[id])
                        scores.append((1-score)*100)
                    columns = 5
                    for i, images in enumerate(img_id):
                        plt.subplot(3,5,i+6)
                        if os.path.isfile("/home/ubuntu/capstone/Data/Flipkart/Tops/"+images):
                            img = image.load_img("/home/ubuntu/capstone/Data/Flipkart/Tops/"+images)
                        elif os.path.isfile("/home/ubuntu/capstone/Data/Flipkart/Hoodie/"+images):
                            img = image.load_img("/home/ubuntu/capstone/Data/Flipkart/Hoodie/" + images)
                        elif os.path.isfile("/home/ubuntu/capstone/Data/Flipkart/Jackets/"+images):
                            img = image.load_img("/home/ubuntu/capstone/Data/Flipkart/Jackets/" + images)
                        x = image.img_to_array(img)
                        plt.imshow(x.astype('uint8'))
                        if ((i+6) == 8):
                            plt.title("Cropped Version\n\n"+img_id[i]+" _ "+str(np.round(scores[i],2))+"%")
                        else:
                            plt.title(img_id[i]+" _ "+str(np.round(scores[i],2))+"%")
                elif clas['name'] == 'Full':
                    distances = scipy.spatial.distance.cdist(full_ex_fea, query_features, "cosine")
                    print(distances.shape)
                    results = zip(range(len(distances)), distances)
                    results = sorted(results, key=lambda x: x[1])
                    print(results[0:5])
                    img_id = []
                    scores = []
                    for id, score in results[0:5]:
                        print(img_ids_full[id])
                        print(score)
                        img_id.append(img_ids_full[id])
                        scores.append((1-score)*100)
                    columns = 5
                    for i, images in enumerate(img_id):
                        plt.subplot(3,5,i+6)
                        if os.path.isfile("/home/ubuntu/capstone/Data/Flipkart/Tops/"+images):
                            img = image.load_img("/home/ubuntu/capstone/Data/Flipkart/Tops/"+images)
                        elif os.path.isfile("/home/ubuntu/capstone/Data/Flipkart/Hoodie/"+images):
                            img = image.load_img("/home/ubuntu/capstone/Data/Flipkart/Hoodie/" + images)
                        elif os.path.isfile("/home/ubuntu/capstone/Data/Flipkart/Jackets/"+images):
                            img = image.load_img("/home/ubuntu/capstone/Data/Flipkart/Jackets/" + images)
                        x = image.img_to_array(img)
                        plt.imshow(x.astype('uint8'))
                        if ((i+6) == 8):
                            plt.title("Cropped Version\n\n"+img_id[i]+" _ "+str(np.round(scores[i],2))+"%")
                        else:
                            plt.title(img_id[i]+" _ "+str(np.round(scores[i],2))+"%")
                print("Full Version")
                img = image.load_img(path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet',pooling='avg')
                query_features = model.predict(x)
                query_features = np.array(query_features).reshape((1,query_features.shape[1]))
                distances = scipy.spatial.distance.cdist(extracted_features, query_features, "cosine")
                print(distances.shape)
                results = zip(range(len(distances)), distances)
                results = sorted(results, key=lambda x: x[1])
                print(results[0:5])
                img_id = []
                scores = []
                for id, score in results[0:5]:
                    print(img_ids[id])
                    print(score)
                    img_id.append(img_ids[id])
                    scores.append((1 - score) * 100)
                columns = 5
                for i, images in enumerate(img_id):
                    plt.subplot(3, 5, i + 11)
                    if os.path.isfile("/home/ubuntu/capstone/Data/Flipkart/Tops/" + images):
                        img = image.load_img("/home/ubuntu/capstone/Data/Flipkart/Tops/" + images)
                    elif os.path.isfile("/home/ubuntu/capstone/Data/Flipkart/Hoodie/" + images):
                        img = image.load_img("/home/ubuntu/capstone/Data/Flipkart/Hoodie/" + images)
                    elif os.path.isfile("/home/ubuntu/capstone/Data/Flipkart/Jackets/" + images):
                        img = image.load_img("/home/ubuntu/capstone/Data/Flipkart/Jackets/" + images)
                    x = image.img_to_array(img)
                    plt.imshow(x.astype('uint8'))
                    if ((i + 11) == 13):
                        plt.title("Full Version\n\n" + img_id[i] + " _ " + str(np.round(scores[i], 2)) + "%")
                    else:
                        plt.title(img_id[i] + " _ " + str(np.round(scores[i], 2)) + "%")
                fig1 = plt.gcf()
                plt.show()
                plt.draw()
                fig1.savefig('Inception_res_ex15.jpg', dpi=100)
