# Load the necessary Files
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import time


start_time = time.time()
#load the features
tops_features_incep = np.load("/home/ubuntu/capstone/final_inception_features/Upper_InceptionResnet_features.npy")
#Standardize it
standarized_data_incep = StandardScaler().fit_transform(tops_features_incep)
print(standarized_data_incep.shape)
#load tnse model
t_model = TSNE(n_components=2, random_state=42,perplexity=50, n_iter=250)
tsne_data_incep = t_model.fit_transform(standarized_data_incep[:2000])
# Plot the distribution
plt.scatter(tsne_data_incep[:,0],tsne_data_incep[:,1])
plt.title("Inception-Resnet TNSE PLOT")
plt.xlabel("incep_x")
plt.ylabel("incep_y")
plt.savefig("Inception_tnse.png")
plt.show()
print("End of Inception", (time.time() - start_time))

start_time = time.time()
#load the features
tops_features_vgg16 = np.load("/home/ubuntu/capstone/Flipkart_Upper_VGG16_features_avg.npy")
#Standardize it
standarized_data_vgg16 = StandardScaler().fit_transform(tops_features_vgg16)
print(standarized_data_vgg16.shape)
#load tnse model
t_model_vgg = TSNE(n_components=2, random_state=42,perplexity=50, n_iter=250)
tsne_data_vgg16 = t_model_vgg.fit_transform(standarized_data_vgg16[:2000])
plt.scatter(tsne_data_vgg16[:,0],tsne_data_vgg16[:,1])
plt.title("VGG TNSE PLOT")
plt.xlabel("VGG_x")
plt.ylabel("VGG_y")
plt.savefig("VGG16_tnse.png")
plt.show()

print("End of VGG16", (time.time() - start_time))

start_time = time.time()
#load the features
tops_features_res = np.load("/home/ubuntu/capstone/Flipkart_Upper_Resnet_features_avg.npy")
#Standardize it
standarized_data_res = StandardScaler().fit_transform(tops_features_res)
print(standarized_data_res.shape)
#load tnse model
t_model_res = TSNE(n_components=2, random_state=42,perplexity=50, n_iter=250)
tsne_data_res = t_model_res.fit_transform(standarized_data_res[:2000])
#load tnse model
plt.scatter(tsne_data_res[:,0],tsne_data_res[:,1])
plt.title("Resnet TNSE PLOT")
plt.xlabel("resnet_x")
plt.ylabel("resnet_y")
plt.savefig("Resnet_tnse.png")
plt.show()

print("End of Resnet", (time.time() - start_time))

start_time = time.time()
#load the features
tops_features_eff = np.load("/home/ubuntu/capstone/Flipkart_Upper_Efficient_features_avg.npy")
#Standardize it
standarized_data_eff = StandardScaler().fit_transform(tops_features_eff)
print(standarized_data_eff.shape)
#load tnse model
t_model_eff = TSNE(n_components=2, random_state=42,perplexity=50, n_iter=250)
tsne_data_eff = t_model_eff.fit_transform(standarized_data_eff[:2500])
#load tnse model
plt.scatter(tsne_data_eff[:,0],tsne_data_eff[:,1])
plt.title("Efficient Net TNSE PLOT")
plt.xlabel("effi_net_x")
plt.ylabel("effi_net_y")
plt.savefig("Efficient_tnse.png")
plt.show()

print("End of Efficient", (time.time() - start_time))
'''
start_time = time.time()

tops_features_incep = np.load("./incep.npy")
standarized_data_incep = StandardScaler().fit_transform(tops_features_incep)
print(standarized_data_incep.shape)
t_model = TSNE(n_components=2, random_state=42,perplexity=50, n_iter=250)
tsne_data_incep = t_model.fit_transform(standarized_data_incep[:2490])
plt.scatter(tsne_data_incep[:,0],tsne_data_incep[:,1])
plt.show()

print("End of incep", (time.time() - start_time))


fig,ax = plt.subplots(3,2)
ax[0,0].scatter(tsne_data[:,0],tsne_data[:,1])
ax[0,0].set_xlabel("MobileNet_x")
ax[0,0].set_ylabel("MobileNet_y")

ax[0,1].scatter(tsne_data_res[:,0],tsne_data_res[:,1])
ax[0,1].set_xlabel("ResNet_x")
ax[0,1].set_ylabel("ResNet_y")

ax[1,0].scatter(tsne_data_dense[:,0],tsne_data_dense[:,1])
ax[1,0].set_xlabel("DenseNet_X")
ax[1,0].set_ylabel("DenseNet_Y")

ax[1,1].scatter(tsne_data_effi[:,0],tsne_data_effi[:,1])
ax[1,1].set_xlabel("EfficientNet_X")
ax[1,1].set_ylabel("EfficientNet_Y")

ax[2,0].scatter(tsne_data_vgg16[:,0],tsne_data_vgg16[:,1])
ax[2,0].set_xlabel("VGG16_X")
ax[2,0].set_ylabel("VGG16_Y")

ax[2,1].scatter(tsne_data_vgg19[:,0],tsne_data_vgg19[:,1])
ax[2,1].set_xlabel("VGG19_X")
ax[2,1].set_ylabel("VGG19_Y")


fig.suptitle('t-SNE Feature extraction plots', fontsize=16)
fig.savefig('feature_extraction_cropped.png')

plt.show()

'''



