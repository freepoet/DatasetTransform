# -*- coding: utf-8 -*-
"""
@File    : test02.py
@Time    : 1/15/21 4:16 PM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
1/15/21 4:16 PM      1.0         None
# @Software: PyCharm
"""

import os
import xml.dom.minidom
import cv2 as cv

ImgPath = '../data/SSDD/VOC2007/JPEGImages/'
AnnoPath = '../data/SSDD/VOC2007/Annotations/'  # xml文件地址
save_path = ''
def draw_anchor(ImgPath, AnnoPath, save_path):
    imagelist = os.listdir(ImgPath)  # 乱序的图片名称.jpg  list里单个元素是string类型
    for image in imagelist:
        image_pre, ext = os.path.splitext(image)#将名称和后缀分开来
        imgfile = ImgPath + image #绝对地址
        xmlfile = AnnoPath + image_pre + '.xml'#图片对应的xml文件的绝对地址
        # print(image)
        # 打开xml文档
        DOMTree = xml.dom.minidom.parse(xmlfile)
        # 得到文档元素对象
        collection = DOMTree.documentElement
        # 读取图片
        img = cv.imread(imgfile)#ndarray格式

        filenamelist = collection.getElementsByTagName("filename")
        filename = filenamelist[0].childNodes[0].data
        # print(filename)
        # 得到标签名为object的信息
        objectlist = collection.getElementsByTagName("object")#可能含有多个object

        for objects in objectlist:
            # 每个object中得到子标签名为name的信息
            namelist = objects.getElementsByTagName('name')
            # 通过此语句得到具体的某个name的值
            objectname = namelist[0].childNodes[0].data

            bndbox = objects.getElementsByTagName('bndbox')
            # print(bndbox)
            for box in bndbox:
                x1_list = box.getElementsByTagName('xmin')
                x1 = int(x1_list[0].childNodes[0].data)
                y1_list = box.getElementsByTagName('ymin')
                y1 = int(y1_list[0].childNodes[0].data)
                x2_list = box.getElementsByTagName('xmax')  # 注意坐标，看是否需要转换
                x2 = int(x2_list[0].childNodes[0].data)
                y2_list = box.getElementsByTagName('ymax')
                y2 = int(y2_list[0].childNodes[0].data)
                cv.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
                cv.putText(img, objectname, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0),
                           thickness=2)
                cv.imshow('head', img)
                cv.waitKey(0)  # 这行和下一行必须加上  否则显示有问题
                cv.destroyAllWindows()
                # cv.imwrite(save_path + '/' + filename, img)  # save picture

draw_anchor(ImgPath,AnnoPath,save_path)

