
#Below script is used to fetch the data stored in the google drive
# Reference : https://github.com/circulosmeos/gdown.pl
import os
import gdown
'''
url = 'https://googledrive.com/host/1p2J1iY7C65tyi1whST84b-w_qI2wvVe1'
output = 'data.zip'
gdown.download(url, output, quiet=False)
'''


os.system("git clone https://github.com/circulosmeos/gdown.pl.git")
os.system("cd gdown.pl")
os.system("./gdown.pl https://drive.google.com/file/d/1DI86w1mqYfwq2ffB3GzkIvVyfb7czPTy/edit final_inception_features.zip")

#./gdown.pl https://drive.google.com/file/d/1p2J1iY7C65tyi1whST84b-w_qI2wvVe1/edit data.zip
#Data - 1p2J1iY7C65tyi1whST84b-w_qI2wvVe1 - data.zip
#Feature Extraction VGG - 1AL-kIrTNRDx9m48KovcSfWHTuGEAjRZi - feature_extraction_vgg.zip
#Feature Extraction cropped - 1fUnsXHl8QOq1qIDUdTxo9T8PPgDUBoYO - feature_extraction_cropped.zip
#Similarity features - 1PDw6jyIK9vEeocYGqX65dUTuVURDLQU5 - similarity_features.zip
#Feature Extraction - 1zgNYXn73jZJLgKcU06h0yjKqsrX8UV3m -feature_extraction.zip