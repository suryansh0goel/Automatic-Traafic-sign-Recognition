import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import random
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sb

model = load_model('tsdr_model')

test = pd.read_csv('D:\\GTSRB dataset\\Test.csv')
paths = test['Path'].values
classid = test['ClassId'].values

sample_img = []
sample_img1 = []
for path in paths:
    img = cv2.imread('D:\\GTSRB dataset\\'+path)
    img = cv2.resize(img, (150, 150))
    img_ = cv2.resize(img,(32,32))
    sample_img.append(img)
    sample_img1.append(img_)

sample_img = np.array(sample_img)
sample_img1 = np.array(sample_img1)
prediction = model.predict(sample_img1)
predicted_class = np.argmax(prediction, axis=-1)
# c_m = metrics.confusion_matrix(classid,predicted_class)
# print(metrics.classification_report(classid,predicted_class))
# sb.heatmap(c_m,annot=True)
# print("Test Accuracy :  "+str(accuracy_score(classid,predicted_class)))
# print("Precision :  "+str(metrics.precision_score(classid,predicted_class,average='macro')))
# print("Recall :  "+str(metrics.recall_score(classid,predicted_class,average='macro')))
# print("F1 score :  "+str(metrics.f1_score(classid,predicted_class,average='macro')))
# plt.show()

classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',
            3:'Speed limit (50km/h)',
            4:'Speed limit (60km/h)',
            5:'Speed limit (70km/h)',
            6:'Speed limit (80km/h)',
            7:'End of speed limit (80km/h)',
            8:'Speed limit (100km/h)',
            9:'Speed limit (120km/h)',
            10:'No passing',
            11:'No passing veh over 3.5 tons',
            12:'Right-of-way at intersection',
            13:'Priority road',
            14:'Yield',
            15:'Stop',
            16:'No vehicles',
            17:'Veh > 3.5 tons prohibited',
            18:'No entry',
            19:'General caution',
            20:'Dangerous curve left',
            21:'Dangerous curve right',
            22:'Double curve',
            23:'Bumpy road',
            24:'Slippery road',
            25:'Road narrows on the right',
            26:'Road work',
            27:'Traffic signals',
            28:'Pedestrians',
            29:'Children crossing',
            30:'Bicycles crossing',
            31:'Beware of ice/snow',
            32:'Wild animals crossing',
            33:'End speed + passing limits',
            34:'Turn right ahead',
            35:'Turn left ahead',
            36:'Ahead only',
            37:'Go straight or right',
            38:'Go straight or left',
            39:'Keep right',
            40:'Keep left',
            41:'Roundabout mandatory',
            42:'End of no passing',
            43:'End no passing veh > 3.5 tons' }

font = cv2.FONT_HERSHEY_SIMPLEX
org = (5, 25)
fontScale = 0.4
color = (255, 0, 0)
thickness = 1
r = random.randint(0, len(prediction))

for i in range(r,r+200):
    if(predicted_class[i]!=classid[i]):
        x_axis = sorted(prediction[i],reverse=True)[0:5]
        y_axis = []
        for j in x_axis:
            y_axis.append(classes[list(prediction[i]).index(j)+1])
        x_axis.reverse()
        y_axis.reverse()
        plt.subplot(2,1,2)
        plt.barh(y_axis[0:4],x_axis[0:4],0.4,color='g')
        plt.barh(y_axis[4], x_axis[4], 0.4,color='r')
        plt.subplot(2,1,1)
        yellow_box = cv2.imread('yellow_box_img.jpg')
        yellow_box = cv2.resize(yellow_box, (150, 50))
        image = cv2.putText(yellow_box, classes[predicted_class[i]+1], org, font, fontScale, color, thickness, cv2.LINE_AA)
        image = np.concatenate((sample_img[i],image))
        plt.imshow(image)
        plt.savefig('D:\\GTSRB dataset\\Testing2\\'+str(i-r),bbox_inches='tight')
        plt.clf()
        # plt.show()










