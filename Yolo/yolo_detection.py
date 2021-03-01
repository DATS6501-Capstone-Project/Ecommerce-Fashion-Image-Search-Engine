import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

os.chdir("./check_new/darknet")
test_image = 'data/obj/TUNEHYGZXC4FMSTE.jpg'
command_darknet = "./darknet detector test data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_final.weights -dont_show "+test_image
os.system(command_darknet)
plt.imshow(mpimg.imread('predictions.jpg'))
plt.show()