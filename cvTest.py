import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
name = '/home/manoj/Documents/SDC/CarND-Behavioral-Cloning-P3/examples/sampleImage.jpg'
#img = cv2.imread(name)
#imgrgb= img#cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#print(img)
#print(np.array(img).size)
##imgrgb = imgrgb[51:135,:]
#plt.figure()
#plt.title('Plain rgb image')
#plt.imshow(imgrgb)
#plt.show()
#
#
#imgrgb = imgrgb[51:135,:]
#plt.figure()
#plt.title('cropped Plain rgb image')
#plt.imshow(imgrgb)
#plt.show()
#
#
#
#plt.savefig('./examples/flippedImage.jpg')
#
plt.figure()
img = Image.open(name)
#img = img.crop((0, 51, 320, 135))
#img = img.resize((200,66), Image.ANTIALIAS)
plt.title('resized rgb image')
plt.imshow(img)
plt.show()


plt.figure()
img = Image.open(name)
img = img.crop((0, 51, 320, 135))
#img = img.resize((200,66), Image.ANTIALIAS)
plt.title('resized rgb image')
plt.imshow(img)
plt.show()


plt.figure()
img = Image.open(name)
img = img.crop((0, 51, 320, 135))
img = img.resize((200,66), Image.ANTIALIAS)
plt.title('resized rgb image')
plt.imshow(img)
plt.show()

print(np.array(img).size)
#
#print(np.array(imgrgb).size)


