# -*- coding: utf-8 -*-
# @Time    : 2019/9/29 0029 8:41
# @Author  : 
# @FileName: aaa.py
# @Software: PyCharm

import matplotlib.pylab as plt
import numpy as np
import cv2
def load_images():
    # for i in range(10):
    #     for j in range(1,3):
    #         try:
    #             image = cv2.imread('./images1/%d-%d.PNG'%(i, j))
    #             image = cv2.resize(image, dsize=(28,28))
    #             image = image[:, :, [2, 1, 0]]
    #             image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #             cv2.imwrite('./images1/%d-%d.PNG'%(i, j), image_gray)
    #
    #         except:
    #             continue
    img = []
    img_labels = []
    for i in range(10):
        for j in range(1, 3):
            try:
                image1 = cv2.imread('./images1/%d-%d.PNG'%(i, j))
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                image1 = image1.reshape((-1, 28*28))
                labels = load_images_labels()
                img.append(255-image1)
                img_labels.append(labels[i])
                # print(image1.shape)
            except:
                continue
    return np.array(img).reshape(-1, 28*28), np.array(img_labels)

# images = load_images()
def load_images_labels():
    labels = np.diag([1]*10)
    return labels

if __name__ == '__main__':
    images_data, img_labs = load_images()
    print(images_data)
