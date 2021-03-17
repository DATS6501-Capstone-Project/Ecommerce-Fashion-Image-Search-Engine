import scipy.spatial
from keras import applications
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np


extracted_features = np.load('./Flipkart_Tops_ResNet_features.npy')
img_ids = np.load('./Flipkart_Tops_ResNet_feature_img_ids.npy')

path = "/home/ubuntu/capstone/Data/Flipkart/Tops/TUNEG39HT7DZCGZQ.jpg"
model = applications.ResNet50(include_top=False, weights='imagenet')
img = image.load_img(path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
query_features = model.predict(x)
query_features = np.array(query_features).reshape((1,100352))
print(query_features)
print(extracted_features)


distances = scipy.spatial.distance.cdist(extracted_features, query_features, "cosine")
print(distances.shape)
results = zip(range(len(distances)), distances)
results = sorted(results, key=lambda x: x[1])
print(results[0:5])
for id,score in results[0:5]:
    print(img_ids[id])
    print(score)