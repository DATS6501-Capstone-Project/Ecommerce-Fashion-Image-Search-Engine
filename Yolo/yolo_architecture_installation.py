import os
import gdown

print("Darknet Architecture installation \n")
#print("Is darknet repo is already cloned? (Y/N)")
repo = str(input("Is darknet repo is already cloned? (Y/N)"))
if repo == "N":
    os.chdir("./check_new")
    os.system("git clone https://github.com/AlexeyAB/darknet.git")
    print("Choose number of label category types - (3/9)")
    cat = int(input())
    if cat == 3:
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
    elif cat == 9:
        url = 'https://drive.google.com/uc?id=19w4NNjsIkLtdLPPSqOIjet30EbH1Sa5o'
        output = 'cat_files.zip'
        gdown.download(url, output, quiet=False)
        os.system("unzip cat_files.zip")
        os.system("cp ./cat_files/yolo-obj.cfg ./darknet/cfg")
        os.system("cp ./cat_files/obj.data ./darknet/data")
        os.system("cp ./cat_files/obj.names ./darknet/data")
        os.system("cp ./files/yolo-obj_final.weights ./darknet")
        os.system("cp ./cat_files/Makefile ./darknet")
        os.chdir("./darknet")
        os.system("pwd")
        os.system("pip install --upgrade cmake")
        os.system("sudo apt remove --purge cmake")
        os.system("hash -r")
        os.system("sudo snap install cmake --classic")
        os.system("./build.sh")
        os.system("make")
    else:
        print("Entered category is invalid")
elif repo == "Y":
    os.chdir("./check_new")
    print("Need to add data files - (Y/N)")
    dats = str(input())
    if dats == 'Y':
        print("Choose number of label category types - (3/9)")
        cat = int(input())
        if cat == 3:
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
        elif cat == 9:
            url = 'https://drive.google.com/uc?id=19w4NNjsIkLtdLPPSqOIjet30EbH1Sa5o'
            output = 'cat_files.zip'
            gdown.download(url, output, quiet=False)
            os.system("unzip cat_files.zip")
            os.system("cp ./cat_files/yolo-obj.cfg ./darknet/cfg")
            os.system("cp ./cat_files/obj.data ./darknet/data")
            os.system("cp ./cat_files/obj.names ./darknet/data")
            os.system("cp ./files/yolo-obj_final.weights ./darknet")
            os.system("cp ./cat_files/Makefile ./darknet")
            os.chdir("./darknet")
            os.system("pwd")
            os.system("pip install --upgrade cmake")
            os.system("sudo apt remove --purge cmake")
            os.system("hash -r")
            os.system("sudo snap install cmake --classic")
            os.system("./build.sh")
            os.system("make")
        else:
            print("Entered category is invalid")
    elif dats == 'N':
        print("Object detection model is ready to be used")
    else:
        print("Entered value is invalid")
else:
    print("Entered Value is invalid")