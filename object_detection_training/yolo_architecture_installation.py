# Reference - Yolov4 - AlexeyAB darknet repositories
# Reference - https://github.com/AlexeyAB/darknet
# Yolov4 Darknet version is used here
# Initial setting to cfg files are made  as per instructions present in the AlexeyAB github pages
# Below Linux command is ran on the .sh files for linux setup. For python user, we have added as python script. Its always advisable to run each command separtely in the linux server

import os
import gdown

print("Darknet Architecture installation \n")
#print("Is darknet repo is already cloned? (Y/N)")
repo = str(input("Is darknet repo is already cloned? (Y/N)"))
if repo == "N":
    os.chdir("./check_new")
    os.system("git clone https://github.com/AlexeyAB/darknet.git")
    url = 'https://drive.google.com/uc?id=1G795Dki1wR7Ixpp_DRrbYK05H0xB17UT'
    output = 'files.zip'
    gdown.download(url, output, quiet=False)
    os.system("unzip files.zip")
    os.system("cp ./files/yolo-obj.cfg ./darknet/cfg")
    os.system("cp ./files/obj.data ./darknet/data")
    os.system("cp ./files/obj.names ./darknet/data")
    os.system("cp ./files/yolo-obj_final.weights ./darknet")
    os.system("cp ./files/Makefile ./darknet")
    os.chdir("./darknet")
    os.system("pwd")
    os.system("pip install --upgrade cmake")
    os.system("sudo apt remove --purge cmake")
    os.system("hash -r")
    os.system("sudo snap install cmake --classic")
    os.system("./build.sh")
    os.system("make")
    print("Do you want to train custom model (Y/N)? (If you want to test with existing weight , please N)")
    train = str(input())
    if train == "Y":
        print("Make sure you added the train data images under data/train folder")
        os.system("darknet.exe partial cfg/yolov4-tiny-custom.cfg yolov4-tiny.weights yolov4-tiny.conv.29 29")
        os.system("darknet.exe detector train data/obj.data yolov4-tiny-obj.cfg yolov4-tiny.conv.29")
    elif train == "N":
        print("Initial setting is completed. You can start testing")
elif repo == "Y":
    os.chdir("./check_new")
    print("Need to add data files - (Y/N)")
    dats = str(input())
    if dats == 'Y':
        url = 'https://drive.google.com/uc?id=1G795Dki1wR7Ixpp_DRrbYK05H0xB17UT'
        output = 'files.zip'
        gdown.download(url, output, quiet=False)
        os.system("unzip files.zip")
        os.system("cp ./files/yolo-obj.cfg ./darknet/cfg")
        os.system("cp ./files/obj.data ./darknet/data")
        os.system("cp ./files/obj.names ./darknet/data")
        os.system("cp ./files/yolo-obj_final.weights ./darknet")
        os.system("cp ./files/Makefile ./darknet")
        os.chdir("./darknet")
        os.system("pwd")
        os.system("pip install --upgrade cmake")
        os.system("sudo apt remove --purge cmake")
        os.system("hash -r")
        os.system("sudo snap install cmake --classic")
        os.system("./build.sh")
        os.system("make")
        print("Do you want to train custom model (Y/N)? (If you want to test with existing weight , please N)")
        train = str(input())
        if train == "Y":
            print("Make sure you added the train data images under data/train folder")
            os.system("darknet.exe partial cfg/yolov4-tiny-custom.cfg yolov4-tiny.weights yolov4-tiny.conv.29 29")
            os.system("darknet.exe detector train data/obj.data yolov4-tiny-obj.cfg yolov4-tiny.conv.29")
        elif train == "N":
            print("Initial setting is completed. You can start testing")
    elif dats == 'N':
        print("Object detection model is ready to be used")
    else:
        print("Entered value is invalid")
else:
    print("Entered Value is invalid")