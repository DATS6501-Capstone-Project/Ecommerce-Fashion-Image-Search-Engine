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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

'''
extracted_features = np.load('./similarity_features/Flipkart_Upper_InceptionResnet_features.npy')
img_ids = np.load('./similarity_features/Flipkart_Upper_InceptionResnet_feature_img_ids.npy')
print(extracted_features.shape)
'''

upper_ex_fea = np.load('./final_inception_features/Upper_InceptionResnet_features.npy')
img_ids_upp = np.load('./final_inception_features/Upper_InceptionResnet_feature_img_ids.npy')
'''
#pca.explained_variance_ratio_
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(upper_ex_fea)
# Apply transform to both the training set and the test set.
train_img = scaler.transform(upper_ex_fea)
pca = PCA(0.99)
pca.fit(train_img)
print(pca.n_components_)
trans = pca.transform(train_img)
upper_ex_fea = trans
'''
lower_ex_fea = np.load('./final_inception_features/Lower_InceptionResnet_features.npy')
img_ids_low = np.load('./final_inception_features/Lower_InceptionResnet_feature_img_ids.npy')
full_ex_fea = np.load('./final_inception_features/Full_InceptionResnet_features.npy')
img_ids_full = np.load('./final_inception_features/Full_InceptionResnet_feature_img_ids.npy')
path = "/home/ubuntu/capstone/long_img/white_long.jpg"
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
            if clas['confidence'] >= 0.60:
                predict_image = Image.open(path)
                plt.figure(figsize=(20, 10))
                plt.subplot(3, 5, 3)
                img = image.load_img(path)
                x = image.img_to_array(img)
                plt.imshow(x.astype('uint8'))
                plt.title("Inception ResNet \n\n Example 10 \n\n Original Image")
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
                np.savetxt('test.out', query_features, delimiter=',')
                print(clas['name'])
                print("Cropped Version")
                print(query_features)
                #query_features = pca.transform(query_features)
                if clas['name'] == 'Upper':
                    distances = scipy.spatial.distance.cdist(upper_ex_fea, query_features, "cosine")
                    results = zip(range(len(distances)), distances)
                    results = sorted(results, key=lambda x: x[1])
                    res = pd.DataFrame(results, columns=['id','sco'])
                    res_ids = res.apply(lambda x:x['id'], axis=1)
                    pd.DataFrame(img_ids_upp[res['id']]).to_csv("results_purple.csv", index=False)
                    print(results[0:5])
                    img_id = []
                    scores = []
                    uni_id = []
                    unique_list = []
                    for id, score in results[0:25]:
                        print(img_ids_upp[id])
                        print(score)
                        img_id.append(img_ids_upp[id])
                        values = img_ids_upp[id].split("/")[-1].split("_")[0].split(".")[0]
                        uni_id.append("/".join(img_ids_upp[id].split("/")[:-1])+"/"+values+".jpg")
                        scores.append((1-score)*100)
                    columns = 5
                    for li in uni_id:
                        if li not in unique_list:
                            unique_list.append(li)
                    for i, images in enumerate(unique_list[:5]):
                        plt.subplot(3,5,i+6)
                        images1 = img_id[uni_id.index(images)]
                        if os.path.isfile(images):
                            img = image.load_img(images)
                        x = image.img_to_array(img)
                        plt.imshow(x.astype('uint8'))
                        if ((i+6) == 8):
                            plt.title("Cosine Similarity\n\n"+uni_id[uni_id.index(images)].split("/")[-1]+"\n"+str(np.round(scores[uni_id.index(images)],2))+"%")
                        else:
                            plt.title(uni_id[uni_id.index(images)].split("/")[-1]+"\n"+str(np.round(scores[uni_id.index(images)],2))+"%")
                    print("\n Euclidean Evaluation Results")
                    distances = scipy.spatial.distance.cdist(upper_ex_fea, query_features, "euclidean")
                    results = zip(range(len(distances)), distances)
                    results = sorted(results, key=lambda x: x[1])
                    res = pd.DataFrame(results, columns=['id','sco'])
                    res_ids = res.apply(lambda x:x['id'], axis=1)
                    print(results[0:5])
                    img_id = []
                    scores = []
                    uni_id = []
                    unique_list = []
                    for id, score in results[0:25]:
                        print(img_ids_upp[id])
                        print(score)
                        img_id.append(img_ids_upp[id])
                        values = img_ids_upp[id].split("/")[-1].split("_")[0].split(".")[0]
                        uni_id.append("/".join(img_ids_upp[id].split("/")[:-1])+"/"+values+".jpg")
                        scores.append((100-score))
                    columns = 5
                    for li in uni_id:
                        if li not in unique_list:
                            unique_list.append(li)
                    for i, images in enumerate(unique_list[:5]):
                        plt.subplot(3,5,i+11)
                        images1 = img_id[uni_id.index(images)]
                        if os.path.isfile(images):
                            img = image.load_img(images)
                        x = image.img_to_array(img)
                        plt.imshow(x.astype('uint8'))
                        if ((i+11) == 13):
                            plt.title("Euclidean Metric evaluation\n\n"+uni_id[uni_id.index(images)].split("/")[-1]+"\n"+str(np.round(scores[uni_id.index(images)],2))+"%")
                        else:
                            plt.title(uni_id[uni_id.index(images)].split("/")[-1]+"\n"+str(np.round(scores[uni_id.index(images)],2))+"%")
                elif clas['name'] == 'Lower':
                    distances = scipy.spatial.distance.cdist(lower_ex_fea, query_features, "cosine")
                    print(distances.shape)
                    results = zip(range(len(distances)), distances)
                    results = sorted(results, key=lambda x: x[1])
                    print(results[0:5])
                    img_id = []
                    scores = []
                    uni_id = []
                    unique_list = []
                    for id, score in results[0:25]:
                        print(img_ids_low[id])
                        print(score)
                        img_id.append(img_ids_low[id])
                        values = img_ids_low[id].split("/")[-1].split("_")[0].split(".")[0]
                        uni_id.append("/".join(img_ids_low[id].split("/")[:-1]) + "/" + values + ".jpg")
                        scores.append((1-score)*100)
                    columns = 5
                    for li in uni_id:
                        if li not in unique_list:
                            unique_list.append(li)

                    for i, images in enumerate(unique_list[:5]):
                        plt.subplot(3,5,i+6)
                        images1 = img_id[uni_id.index(images)]
                        print(images1)
                        if os.path.isfile(images):
                            img = image.load_img(images1)
                        x = image.img_to_array(img)
                        plt.imshow(x.astype('uint8'))
                        if ((i+6) == 8):
                            plt.title("Cosine Similarity\n\n"+uni_id[uni_id.index(images)].split("/")[-1]+"\n"+str(np.round(scores[uni_id.index(images)],2))+"%")
                        else:
                            plt.title(uni_id[uni_id.index(images)].split("/")[-1]+"\n"+str(np.round(scores[uni_id.index(images)],2))+"%")
                    print("\n Euclidean Evaluation Results")
                    distances = scipy.spatial.distance.cdist(lower_ex_fea, query_features, "euclidean")
                    print(distances.shape)
                    results = zip(range(len(distances)), distances)
                    results = sorted(results, key=lambda x: x[1])
                    print(results[0:5])
                    img_id = []
                    scores = []
                    uni_id = []
                    unique_list = []
                    for id, score in results[0:25]:
                        print(img_ids_low[id])
                        print(score)
                        img_id.append(img_ids_low[id])
                        values = img_ids_low[id].split("/")[-1].split("_")[0].split(".")[0]
                        uni_id.append("/".join(img_ids_low[id].split("/")[:-1]) + "/" + values + ".jpg")
                        scores.append((100-score))
                    columns = 5
                    for li in uni_id:
                        if li not in unique_list:
                            unique_list.append(li)

                    for i, images in enumerate(unique_list[:5]):
                        plt.subplot(3,5,i+11)
                        images1 = img_id[uni_id.index(images)]
                        if os.path.isfile(images):
                            img = image.load_img(images1)
                        x = image.img_to_array(img)
                        plt.imshow(x.astype('uint8'))
                        if ((i+11) == 13):
                            plt.title("Euclidean Metric evaluation\n\n"+uni_id[uni_id.index(images)].split("/")[-1]+"\n"+str(np.round(scores[uni_id.index(images)],2))+"%")
                        else:
                            plt.title(uni_id[uni_id.index(images)].split("/")[-1]+"\n"+str(np.round(scores[uni_id.index(images)],2))+"%")
                elif clas['name'] == 'Full':
                    distances = scipy.spatial.distance.cdist(full_ex_fea, query_features, "cosine")
                    print(distances.shape)
                    results = zip(range(len(distances)), distances)
                    results = sorted(results, key=lambda x: x[1])
                    print(results[0:5])
                    img_id = []
                    scores = []
                    uni_id = []
                    unique_list = []
                    for id, score in results[0:25]:
                        print(img_ids_full[id])
                        print(score)
                        img_id.append(img_ids_full[id])
                        values = img_ids_full[id].split("/")[-1].split("_")[0].split(".")[0]
                        uni_id.append("/".join(img_ids_full[id].split("/")[:-1]) + "/" + values + ".jpg")
                        scores.append((1-score)*100)
                    columns = 5
                    for li in uni_id:
                        if li not in unique_list:
                            unique_list.append(li)
                    a = unique_list[:3]
                    a.extend(unique_list[3:5])
                    for i, images in enumerate(a):
                        plt.subplot(3,5,i+6)
                        images1 = img_id[uni_id.index(images)]
                        if os.path.isfile(images):
                            img = image.load_img(images)
                        x = image.img_to_array(img)
                        plt.imshow(x.astype('uint8'))
                        if ((i+6) == 8):
                            plt.title("Cosine Similarity\n\n"+uni_id[uni_id.index(images)].split("/")[-1]+"\n"+str(np.round(scores[uni_id.index(images)],2))+"%")
                        else:
                            plt.title(uni_id[uni_id.index(images)].split("/")[-1]+"\n"+str(np.round(scores[uni_id.index(images)],2))+"%")
                    print("\n Euclidean Evaluation Results")
                    distances = scipy.spatial.distance.cdist(full_ex_fea, query_features, "euclidean")
                    print(distances.shape)
                    results = zip(range(len(distances)), distances)
                    results = sorted(results, key=lambda x: x[1])
                    print(results[0:5])
                    img_id = []
                    scores = []
                    uni_id = []
                    unique_list = []
                    for id, score in results[0:25]:
                        print(img_ids_full[id])
                        print(score)
                        img_id.append(img_ids_full[id])
                        values = img_ids_full[id].split("/")[-1].split("_")[0].split(".")[0]
                        uni_id.append("/".join(img_ids_full[id].split("/")[:-1]) + "/" + values + ".jpg")
                        scores.append((100-score))
                    columns = 5
                    for li in uni_id:
                        if li not in unique_list:
                            unique_list.append(li)
                    a = unique_list[:2]
                    a.extend(unique_list[2:4])
                    a.extend(unique_list[4:5])
                    for i, images in enumerate(a):
                        plt.subplot(3,5,i+11)
                        images1 = img_id[uni_id.index(images)]
                        if os.path.isfile(images):
                            img = image.load_img(images)
                        x = image.img_to_array(img)
                        plt.imshow(x.astype('uint8'))
                        if ((i+11) == 13):
                            plt.title("Euclidean Metric evaluation\n\n"+uni_id[uni_id.index(images)].split("/")[-1]+"\n"+str(np.round(scores[uni_id.index(images)],2))+"%")
                        else:
                            plt.title(uni_id[uni_id.index(images)].split("/")[-1]+"\n"+str(np.round(scores[uni_id.index(images)],2))+"%")
                fig1 = plt.gcf()
                plt.show()
                plt.draw()
                fig1.savefig('./new_test_results/full_samples/white_long.jpg', dpi=100)
'''
distances_ans = 1 - distances
df_cls = pd.DataFrame({'Clusters': clus, 'Similarity_Score': distances_ans.reshape((distances_ans.shape[0]))})
agg_val = {}
agg_val['Clusters'] = 'first'
agg_val['Similarity_Score'] = 'mean'
Clus_score = df_cls[['Clusters', 'Similarity_Score']].groupby('Clusters').agg(agg_val)
Clus_score = Clus_score.reset_index(drop=True)
Clus_score = Clus_score.sort_values(['Similarity_Score'], ascending=False)
print(Clus_score)
max_cluster = Clus_score.iloc[0]['Clusters']
print(max_cluster)
max_scores_li = pd.Index(df_cls[df_cls['Clusters'] == max_cluster]['Similarity_Score'])
print(max_scores_li.argmax())
print(img_ids_upp[max_scores_li.argmax()])
'''
