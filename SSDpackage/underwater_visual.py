#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from cv2 import imread
import sys
from SSDpackage.ssd_structure import SSD300
from SSDpackage.utils_ssd import BBoxUtility
import os
from os.path import basename
import tensorflow.compat.v1 as tf1


# In[2]:


def create_overlay(img, results, plt_fname):
    plt.clf()
    # Parse the outputs.
    det_label = results[:, 0]
    det_conf = results[:, 1]
    det_xmin = results[:, 2]
    det_ymin = results[:, 3]
    det_xmax = results[:, 4]
    det_ymax = results[:, 5]

    # Get detections with confidence higher than 0.6.
    person_indices = [i for i, label in enumerate(det_label) if label == 15]
    person_conf = det_conf[person_indices]
    top_indices = [i for i, conf in enumerate(person_conf) if conf >= 0.6 ]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    plt.imshow(img / 255.)
    currentAxis = plt.gca()
    currentAxis.axis('off')

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = 15
            #label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}'.format(score)
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords,
                                            fill=False,
                                            edgecolor=color,
                                            linewidth=2))
        currentAxis.text(xmin, ymin, display_txt,
                         bbox={'facecolor': color, 'alpha': 0.5})
        
    plt.savefig(plt_fname)
    print("save "+plt_fname)


# In[3]:


def person_detector(img):
    voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                   'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                   'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
                   'Sheep', 'Sofa', 'Train', 'Tvmonitor']
    NUM_CLASSES = len(voc_classes) + 1
    input_shape = (300, 300, 3)
    model = SSD300(input_shape, num_classes=NUM_CLASSES)
    model.load_weights('SSDpackage\weights_SSD300.hdf5', by_name=True)
    bbox_util = BBoxUtility(NUM_CLASSES)
     # Load the inputs
    inputs = []
    images = []
    img = image.img_to_array(img)
    inputs.append(img.copy())
    # 前置處理
    print("前置處理...")
    inputs = preprocess_input(np.array(inputs))

    # 預測
    print("預測...")
    preds = model.predict(inputs, batch_size=1, verbose=1)
    
    # 取得預測結果
    results = bbox_util.detection_out(preds)
    result = results[0]
    
    # Parse the outputs.
    det_label = result[:, 0]
    det_conf = result[:, 1]
    det_xmin = result[:, 2]
    det_ymin = result[:, 3]
    det_xmax = result[:, 4]
    det_ymax = result[:, 5]

    # Get detections with confidence higher than 0.6.
    person_indices = [i for i, label in enumerate(det_label) if label == 15]
    person_conf = det_conf[person_indices]
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    # calculate position
    top_x = (top_xmin+top_xmax)/2
    top_y = (top_ymin+top_ymax)/2
    output = [top_x, top_y, top_conf]
    return [output, result]
    


# In[4]:


if __name__ == "__main__":
    img_path = "SSDpackage\images\diving_4.png"
    output_path = "SSDpackage\images" + "\{}.png".format(basename(os.path.splitext(img_path)[0]))
    picture = image.load_img(img_path, target_size=(300, 300))
    [info1, info2]  = person_detector(picture)
    create_overlay(imread(img_path), info2, output_path)
    print(info1)


# In[ ]:




