import pandas as pd
from sklearn import preprocessing
from PIL import Image
sample_data = pd.read_csv("./sample_train.csv")
sample_test = pd.read_csv("./sample_test.csv")
le = preprocessing.LabelEncoder()
le.fit(sample_data['category_type1'])
print("Classes: ",le.classes_)
sample_data['Category_en'] = le.transform(sample_data['category_type1'])
sample_test['Category_en'] = le.transform(sample_test['category_type1'])
sample_data["label"] = sample_data.apply(lambda x: str(x['Category_en']) +" "+ x['bounding_box'], axis=1)
sample_test["label"] = sample_test.apply(lambda x: str(x['Category_en']) +" "+ x['bounding_box'], axis=1)

for row in sample_data.values:
    filepath = "./"+row[3]
    im = Image.open(filepath)
    width = im.size[0]
    height = im.size[1]
    dw = 1. / width
    dh = 1. / height
    box = row[7].split(" ")
    x = (int(box[1]) + int(box[3]))/2.0
    y = (int(box[2]) + int(box[4]))/2.0
    w = int(box[3]) - int(box[1])
    h = int(box[4]) - int(box[2])
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    label = str(box[0])+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h)
    print(label)
    with open("./"+row[3].split('.')[0]+".txt","wt") as outfile:
        outfile.write(label)
        print("./"+row[3].split('.')[0]+".txt")
    outfile.close()
for row in sample_test.values:
    filepath = "./"+row[3]
    im = Image.open(filepath)
    width = im.size[0]
    height = im.size[1]
    dw = 1. / width
    dh = 1. / height
    box = row[7].split(" ")
    x = (int(box[1]) + int(box[3]))/2.0
    y = (int(box[2]) + int(box[4]))/2.0
    w = int(box[3]) - int(box[1])
    h = int(box[4]) - int(box[2])
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    label = str(box[0])+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h)
    print(label)
    with open("./"+row[3].split('.')[0]+".txt","wt") as outfile:
        outfile.write(label)
        print("./"+row[3].split('.')[0]+".txt")
    outfile.close()


