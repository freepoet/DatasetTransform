# -*- coding: utf-8 -*-
# @Time    : 27/09/2020 08:49
# @Author  : Mingqiang Ning
# @Email   : ningmq_cv@foxmail.com
# @File    : BackgroundRotation.py
# @Software: PyCharm
###   Capture target after rotating background ###
import os

import cv2
from PIL import Image
import xml.etree.ElementTree as ET
import math
import numpy as np
from mmcv import list_from_file

def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1 )
    # cv2.putText(image, text, cordinate, font, size, color,thickness)

def scale_bounding_box(old, img_size, scale_rate):
    # scale_rate need to be higher than 1.0
    xmin, ymin, xmax, ymax = int(old[0]), int(old[1]), int(old[2]), int(old[3])
    delta_x = int((xmax - xmin)*(scale_rate-1) * 0.5)
    delta_y = int((ymax - ymin) * (scale_rate - 1) * 0.5)

    tmp = xmin - delta_x
    if tmp >= 0:
        xmin_new = tmp
    else:
        xmin_new = xmin

    tmp = ymin - delta_y
    if tmp >= 0:
        ymin_new = tmp
    else:
        ymin_new = ymin

    tmp = xmax + delta_x
    if tmp <= img_size[0]-1:
        xmax_new = tmp
    else:
        xmax_new = xmax

    tmp = ymax + delta_y
    if tmp <= img_size[1]-1:
        ymax_new = tmp
    else:
        ymax_new = ymax

    return xmin_new, ymin_new, xmax_new, ymax_new

def ppp(a,b):
    print("#-#-#-#-#-#:::",a,':::',b)

def extract_background(bnd_corner_list, img_size, drop_threshod=0.05):
    """

    :param bnd_corner_list: coordinates of  upper left and bottom right conner point of target(s),
                            data format: [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...], type=list, dtype=int
    :param img_size: image size
                     data format: (width,height), type=tuple, dtype=int
    :param drop_threshod: if length of any side of a backgound is less than int(drop_threshod*img_size), ignore it
    :return: coordinates of  upper left and bottom right conner point of backgrounds,
             data format: [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...], type=list, dtype=int
    """
    w,h = img_size[0],img_size[1]
    w_drop, h_drop = int(drop_threshod*w), int(drop_threshod*h)
    bgd = []
    # if there is no object, divide picture into 4 parts
    if len(bnd_corner_list) == 0:
        w_middle, h_middle = int(w/2), int(h/2)
        tmp = [0, 0, w_middle-1, h_middle-1]  # upper left rectangle
        bgd.append(tmp)
        tmp = [w_middle,0, w-1, h_middle-1]  # upper right rectangle
        bgd.append(tmp)
        tmp = [0, h_middle, w_middle-1, h-1]  # bottom left rectangle
        bgd.append(tmp)
        tmp = [w_middle, h_middle, w-1, h-1]  # bottom right rectangle
        bgd.append(tmp)
    # if there is only 1 object in the picture, divide its surroundings into 4 parts
    elif len(bnd_corner_list) == 1:
        ulx,uly,brx,bry = bnd_corner_list[0]  # get corner point of the object
        if ulx >= w_drop:
            tmp = [0, 0, ulx-1, h-1]  # left rectangle
            bgd.append(tmp)
        if brx <= w - w_drop - 1:
            tmp = [brx, 0, w-1, h-1]  # right rectangle
            bgd.append(tmp)
        if uly >= h_drop:
            tmp = [ulx, 0, brx-1, uly-1]  # upper rectangle
            bgd.append(tmp)
        if bry <= h - h_drop - 1:
            tmp = [ulx, bry, brx-1, h-1]  # bottom rectangle
            bgd.append(tmp)
    elif len(bnd_corner_list) > 1:
        # transfer the 2-D list to a 2-D numpy.ndarray
        bnd = np.array(bnd_corner_list)
        # add upper right corner point
        bnd = np.insert(bnd, 2, values=bnd[:,2], axis=1)
        bnd = np.insert(bnd, 3, values=bnd[:, 1], axis=1)
        # add bottom left corner point
        bnd = np.insert(bnd, 4, values=bnd[:, 0], axis=1)
        bnd = np.insert(bnd, 5, values=bnd[:, 6], axis=1)
        # add ID and class to each box
        bnd = np.insert(bnd, 0, values=np.array(range(len(bnd[:,0]))), axis=1)
        bnd = np.insert(bnd, 1, values=np.zeros(len(bnd[:, 0]),dtype = int), axis=1)
        bnd = np.insert(bnd, 6, values=np.array(range(len(bnd[:, 0]))), axis=1)
        bnd = np.insert(bnd, 7, values=np.ones(len(bnd[:, 0]),dtype = int), axis=1)
        # (ID,0,xl,yl,xr,yl,ID,1,xl,yr,xr,yr)
        # ( 0,1,2 ,3, 4, 5,  6,7,8, 9,10,11)
        # get bottom lines and rank it according to their heights
        bottom_line = np.vstack((bnd[:, :6], bnd[:, 6:12]))
        bottom_line_sorted = np.array(sorted(bottom_line, key=lambda x: x[3]))

        # left_line = np.vstack((bnd[:, [0, 1, 4, 5]], bnd[:, [2, 3, 6, 7]]))
        # left_line_sorted = np.array(sorted(left_line, key=lambda x: x[0]))

        # 1. find marco conner and get first batch of background rectangles
        ulx_marco, uly_marco = bnd[:,2:4].min(0)
        brx_marco, bry_marco = bnd[:, 10:12].max(0)
        if ulx_marco >= w_drop:
            tmp = [0, 0, ulx_marco-1, h-1]  # left rectangle
            bgd.append(tmp)
        if brx_marco <= w - w_drop - 1:
            tmp = [brx_marco, 0, w-1, h-1]  # right rectangle
            bgd.append(tmp)
        if uly_marco >= h_drop:
            tmp = [ulx_marco, 0, brx_marco-1, uly_marco-1]  # upper rectangle
            bgd.append(tmp)
        if bry_marco <= h - h_drop - 1:
            tmp = [ulx_marco, bry_marco, brx_marco-1, h-1]  # bottom rectangle
            bgd.append(tmp)
        # 2. extract 2nd batch of background based on a complex rule
        top = []
        top.append([ulx_marco, uly_marco, brx_marco, bry_marco])
        # ppp("bnd_corner_list", bnd_corner_list)
        # ppp("bottom_line_sorted",bottom_line_sorted)
        # ppp("top",top)
        for i in range(len(bottom_line_sorted)-1):
            k = i+1
            bgd_tmp, top_tmp = extract_real_bgd(bottom_line_sorted[k], bnd, top, (w_drop, h_drop))
            if bgd_tmp is not None:
                for i in range(len(bgd_tmp)):
                    bgd.append(bgd_tmp[i])
            if top_tmp is not None:
                for i in range(len(top_tmp)):
                    top.append(top_tmp[0 ])
    # add a cast
    bgd_cast = []
    for j in range(len(bgd)):
        cast_sign = True
        xl_b, yl_b, xr_b, yr_b = bgd[j]
        bgd_small = []
        for i in range(len(bnd_corner_list)):
            # ppp("bnd_corner_list",bnd_corner_list)
            if (xl_b <= bnd_corner_list[i][0]) & (yl_b <=bnd_corner_list[i][1]) & (xr_b >= bnd_corner_list[i][2]) & (yr_b >= bnd_corner_list[i][3]):
                cast_sign = False
                break
        if cast_sign:
            W = xr_b - xl_b
            H = yr_b - yl_b
            min_w_h = min(W, H)
            min_w_h_drop = min(w_drop,h_drop)
            if min_w_h < min_w_h_drop:
                break
            else:
                ratio = (int(W / min_w_h), int(H / min_w_h))
            xl_bgd_small = []
            yl_bgd_small = []
            if ratio[0]>=3:
                delta = int(W/3)
                xl_bgd_small = [xl_b,xl_b+delta,xl_b+delta+delta,xr_b]
                bgd_cast.append([xl_bgd_small[0], yl_b, xl_bgd_small[1], yr_b])
                bgd_cast.append([xl_bgd_small[1], yl_b, xl_bgd_small[2], yr_b])
                bgd_cast.append([xl_bgd_small[2], yl_b, xl_bgd_small[3], yr_b])
            if ratio[1]>=3:
                delta = int(H/3)
                yl_bgd_small = [yl_b,yl_b+delta,yl_b+delta+delta,yr_b]
                bgd_cast.append([xl_b, yl_bgd_small[0], xr_b, yl_bgd_small[1]])
                bgd_cast.append([xl_b, yl_bgd_small[1], xr_b, yl_bgd_small[2]])
                bgd_cast.append([xl_b, yl_bgd_small[2], xr_b, yl_bgd_small[3]])
            if ratio[0]==2:
                delta = int(W/2)
                xl_bgd_small = [xl_b,xl_b+delta,xr_b]
                bgd_cast.append([xl_bgd_small[0], yl_b, xl_bgd_small[1], yr_b])
                bgd_cast.append([xl_bgd_small[1], yl_b, xl_bgd_small[2], yr_b])
                # bgd_cast.append([xl_bgd_small[2], yl_b, xl_bgd_small[3], yr_b])
            if ratio[1]==2:
                delta = int(H/2)
                yl_bgd_small = [yl_b,yl_b+delta,yr_b]
                bgd_cast.append([xl_b, yl_bgd_small[0], xr_b, yl_bgd_small[1]])
                bgd_cast.append([xl_b, yl_bgd_small[1], xr_b, yl_bgd_small[2]])
                # bgd_cast.append([xl_b, yl_bgd_small[2], xr_b, yl_bgd_small[3]])
            if max(ratio)<2:
                bgd_cast.append([xl_b, yl_b, xr_b, yr_b])
    bgd_cast1 = []
    for i in range(len(bgd_cast)):
        xl_b, yl_b, xr_b, yr_b = bgd_cast[i ]
        W = xr_b - xl_b
        H = yr_b - yl_b
        xl_bgd_small = []
        # yl_bgd_small = []
        if W >= w/3:
            delta = int(W / 3)
            xl_bgd_small = [xl_b, xl_b + delta, xl_b + delta + delta, xr_b]
            bgd_cast1.append([xl_bgd_small[0], yl_b, xl_bgd_small[1], yr_b])
            bgd_cast1.append([xl_bgd_small[1], yl_b, xl_bgd_small[2], yr_b])
            bgd_cast1.append([xl_bgd_small[2], yl_b, xl_bgd_small[3], yr_b])
        else:
            bgd_cast1.append([xl_b, yl_b, xr_b, yr_b])

    bgd_cast2 = []
    for i in range(len(bgd_cast1)):
        xl_b, yl_b, xr_b, yr_b = bgd_cast1[i]
        W = xr_b - xl_b
        H = yr_b - yl_b
        # xl_bgd_small = []
        yl_bgd_small = []
        if H >= h/3:
            delta = int(H / 3)
            yl_bgd_small = [yl_b, yl_b + delta, yl_b + delta + delta, yr_b]
            bgd_cast2.append([xl_b, yl_bgd_small[0], xr_b, yl_bgd_small[1]])
            bgd_cast2.append([xl_b, yl_bgd_small[1], xr_b, yl_bgd_small[2]])
            bgd_cast2.append([xl_b, yl_bgd_small[2], xr_b, yl_bgd_small[3]])
        else:
            bgd_cast2.append([xl_b, yl_b, xr_b, yr_b])
    bgd_cast3 = []
    for i in range(len(bgd_cast2)):
        xl_b, yl_b, xr_b, yr_b = bgd_cast2[i]
        if (xr_b>xl_b)&(yr_b>yl_b):
            bgd_cast3.append([xl_b, yl_b, xr_b, yr_b])
    return bgd_cast3
    # return bgd

def extract_real_bgd(bottom_line, bnd, top, drop):
    id, cl, xl, yl, xr, yr = bottom_line
    # judge if current bnd is in other bnds
    bgd_tmp, top_tmp = None, None
    bnd1 = np.delete(bnd, id, 0)  # delete current bnd, bnd1 is used for judging if current bnd is localized inside any other bnd
    # ppp("bnd1",bnd1)
    if judge_not_include(bnd[id,[2,3,10,11]],bnd1):
        # cal_real_bottom
        if cl == 0:
            # ppp("(xl, yl, xr, yr)",(xl, yl, xr, yr))
            include_sign, bottom_tmp, top_tmp = judge_cross0((xl, yl, xr, yr), bnd1, drop, top)
            # ppp("include_sign, bottom_tmp, top_tmp", (include_sign, bottom_tmp, top_tmp))
            if include_sign:
                return None, None
            else:
                bgd_tmp = []
                for i in range(len(bottom_tmp)):
                    # ppp("bottom_tmp[i]",bottom_tmp[i])
                    xl, yl, xr, yr = bottom_tmp[i]
                    y_up = get_y_up((xl, yl, xr, yr), top)
                    # ppp("y_up",y_up)
                    if y_up is not None:
                        bgd_tmp.append([xl,y_up,xr,yr])
        elif cl == 1:
            a = judge_cross1((xl, yl, xr, yr), bnd1, drop, top)
            include_sign=a[0]
            bottom_tmp=a[1]
            top_tmp=a[2]
            if include_sign:
                return None, None
            else:
                bgd_tmp = []
                for i in range(len(bottom_tmp)):
                    xl, yl, xr, yr = bottom_tmp[i]
                    y_up = get_y_up((xl, yl, xr, yr), top)
                    if y_up is not None:
                        bgd_tmp.append([xl, y_up, xr, yr])

    return bgd_tmp, top_tmp

def judge_not_include(tmp,bnd1):
    not_include = True

    xl,yl,xr,yr = tmp
    for i in range(len(bnd1[:,0])):
        if xl>=bnd1[i,2] & yl>=bnd1[i,3] & xr<=bnd1[i,10] & yr<=bnd1[i,11]:
            not_include = False

    return not_include

def judge_cross0(current_bottom, bnd1, drop, top):
    xl, yl, xr, yr = current_bottom
    w_drop, h_drop = drop
    ulx_marco, uly_marco, brx_marco, uly_marco = top[0]
    l=[]
    r=[]
    ml=[]
    mr=[]
    top_tmp=[]
    include_sign = False
    for i in range(len(bnd1[:,0])):
        # Function: bnd1[i,:] represent target polygon, which will be judged to see the cross state with current polygon
        # --------------------------------------------------------------------------------------------------------------
        # judge if the left corner is inside in target polygon
        # update candidate xl through l.append()
        if (xl >= bnd1[i,2]) & (yl >= bnd1[i,3]) & (xl <=bnd1[i,10]) & (yl<=bnd1[i,11]):  # L in
            if (xr >= bnd1[i, 2]) & (yr >= bnd1[i, 3]) & (xr <= bnd1[i, 10]) & (yr <= bnd1[i, 11]):  # R in
                return True, None, None
            else:  # left corner is in and right corner is not in, R not in
                l.append(bnd1[i,10])  # x coordinate of target polygon's right corner is appended as a candidate xl
        # --------------------------------------------------------------------------------------------------------------
        # judge if the right corner is inside in target polygon
        # update candidate xr through r.append()
        if (xr >= bnd1[i, 2]) & (yr >= bnd1[i, 3]) & (xr <= bnd1[i, 10]) & (yr <= bnd1[i, 11]):
            if (xl >= bnd1[i, 2]) & (yl >= bnd1[i, 3]) & (xl <= bnd1[i, 10]) & (yl <= bnd1[i, 11]):
                return True, None, None
            else:  # right corner is in and left corner is not in
                r.append(bnd1[i, 2])  # x coordinate of target polygon's left corner is appended as a candidate xr
        # --------------------------------------------------------------------------------------------------------------
        # judge if both vertical edge of target polygon is crossed with current bottom line
        # update candidate middle point through ml.append() and mr.append()
        if (xl < bnd1[i, 2]) & (xr > bnd1[i, 10]) & (yl > bnd1[i, 3]) & (yr <= bnd1[i, 11]):
            ml.append(bnd1[i, 2])
            mr.append(bnd1[i, 10])

    if len(l) == 0:  # left corner is not in any target polygon, xr---->xl----->real xl through cross method
        # In such condition, top need to be updated
        xl_old = xl
        xl_new = [ulx_marco]
        for i in range(len(bnd1[:, 0])):
            if (bnd1[i,4] < xl_old) & (bnd1[i,3] <= yl) & (bnd1[i,9] >= yl):
                xl_new.append(bnd1[i,4])
        xl = max(xl_new)  # xl is depend on all right edge of target polygon and ulx_marco
        top_tmp.append([xl,yl,xl_old,yr])
    else:
        xl = max(l)

    if len(r) == 0:  # xl---->xr----->real xr through cross method
        # In such condition, top also need to be updated
        xr_old = xr
        xr_new = [brx_marco]
        for i in range(len(bnd1[:, 0])):
            if (bnd1[i,2] > xr_old) & (bnd1[i,3] <= yl) & (bnd1[i,9] >= yl):
                xr_new.append(bnd1[i, 2])
        xr = min(xr_new)  # xr is depend on all left edge of target polygon and brx_marco
        top_tmp.append([xr_old,yr,xr,yr])
    else:
        xr = min(r)

    if len(ml) == 0:
        bottom_tmp = [[xl, yl, xr, yr]]
        return include_sign, bottom_tmp, top_tmp
    elif len(ml) == 1:
        # In this case, we will get two bottom, but it does not affect top_tmp
        # situation can be super complex, so we make a simplification here
        if (xl >= ml[0]) & (xl <= mr[0]):
            xl = mr[0]
        if (xr >= ml[0]) & (xr <= mr[0]):
            xr = ml[0]
        if (xl < ml[0]) & (xr > mr[0]):
            bottom_tmp = [[xl, yl, ml[0], yr],[mr[0], yl, xr, yr]]
        return include_sign, bottom_tmp, top_tmp
    elif len(ml) >1:
        return True, None, None  # simplify the possibility

def judge_cross1(current_bottom, bnd1, drop, top):
    xl, yl, xr, yr = current_bottom
    w_drop, h_drop = drop
    ulx_marco, uly_marco, brx_marco, uly_marco = top[0]
    l=[]
    r=[]
    ml=[]
    mr=[]
    top_tmp=[]
    bottom_tmp = []
    include_sign = False

    for i in range(len(bnd1[:,0])):
        # Function: bnd1[i,:] represent target polygon, which will be judged to see the cross state with current polygon
        # --------------------------------------------------------------------------------------------------------------
        # judge if the left corner is inside in target polygon
        # update candidate xl through l.append()
        ppp("i",i)
        if (xl >= bnd1[i,2]) & (yl >= bnd1[i,3]) & (xl <=bnd1[i,10]) & (yl<=bnd1[i,11]):  # L in
            if (xr >= bnd1[i, 2]) & (yr >= bnd1[i, 3]) & (xr <= bnd1[i, 10]) & (yr <= bnd1[i, 11]):  # R in
                ppp("True, None, None",[True, None, None])
                return True, None, None
            else:  # left corner is in and right corner is not in, R not in
                l.append(bnd1[i,10])  # x coordinate of target polygon's right corner is appended as a candidate xl
        # --------------------------------------------------------------------------------------------------------------
        # judge if the right corner is inside in target polygon
        # update candidate xr through r.append()
        if (xr >= bnd1[i, 2]) & (yr >= bnd1[i, 3]) & (xr <= bnd1[i, 10]) & (yr <= bnd1[i, 11]):
            if (xl >= bnd1[i, 2]) & (yl >= bnd1[i, 3]) & (xl <= bnd1[i, 10]) & (yl <= bnd1[i, 11]):
                ppp("True, None, None", [True, None, None])
                return True, None, None
            else:  # right corner is in and left corner is not in
                r.append(bnd1[i, 2])  # x coordinate of target polygon's left corner is appended as a candidate xr
        # --------------------------------------------------------------------------------------------------------------
        # judge if both vertical edge of target polygon is crossed with current bottom line
        # update candidate middle point through ml.append() and mr.append()
        if (xl < bnd1[i, 2]) & (xr > bnd1[i, 10]) & (yl > bnd1[i, 3]) & (yr <= bnd1[i, 11]):
            ml.append(bnd1[i, 2])
            mr.append(bnd1[i, 10])
    ppp("l",l)
    if len(l) == 0:  # left corner is not in any target polygon, xr---->xl----->real xl through cross method
        # In such condition, top need to be updated
        xl_old = xl
        xl_new = [ulx_marco]
        for i in range(len(bnd1[:, 0])):
            if (bnd1[i,4] < xl_old) & (bnd1[i,3] <= yl) & (bnd1[i,9] >= yl):
                print("i",i)
                print("bnd1[i,8]",bnd1[i,4])
                xl_new.append(bnd1[i,4])
        xl = max(xl_new)  # xl is depend on all right edge of target polygon and ulx_marco
        bottom_tmp.append([xl,yl,xl_old,yr])
    else:
        xl = max(l)
    if len(r) == 0:  # xl---->xr----->real xr through cross method
        # In such condition, top also need to be updated
        xr_old = xr
        xr_new = [brx_marco]
        for i in range(len(bnd1[:, 0])):
            if (bnd1[i,2] > xr_old) & (bnd1[i,3] <= yl) & (bnd1[i,9] >= yl):
                xr_new.append(bnd1[i, 2])
        xr = min(xr_new)  # xr is depend on all left edge of target polygon and brx_marco
        bottom_tmp.append([xr_old,yr,xr,yr])
    else:
        xr = min(r)
    if len(ml) == 0:
        top_tmp = [[xl, yl, xr, yr]]
        return include_sign, bottom_tmp, top_tmp
    elif len(ml) == 1:
        # In this case, we will get two bottom, but it does not affect top_tmp
        # situation can be super complex, so we make a simplification here
        print("xl,ml[0],xl,mr[0]",[xl,ml[0],xl,mr[0]])
        if (xl >= ml[0]) & (xl <= mr[0]):
            xl = mr[0]
        if (xr >= ml[0]) & (xr <= mr[0]):
            xr = ml[0]
        if (xl < ml[0]) & (xr > mr[0]):
            top_tmp = [[xl, yl, ml[0], yr],[mr[0], yl, xr, yr]]

        return include_sign, bottom_tmp, top_tmp
    elif len(ml) >1:
        ppp("True, None, None",[True, None, None])
        return True, None, None  # simplify the possibility

def get_y_up(real_bottom,top):
    xl, yl, xr, yr = real_bottom
    print("real_bottom",real_bottom)
    print("top",top)
    y_up_candidate = []
    for i in range(len(top)):
        if (xl >= top[i][0]) & (xr <= top[i][2]) & (yr > top[i][1]):
            y_up_candidate.append(top[i][1])
    if len(y_up_candidate) == 0:
        y_up = None
    else:
        y_up = max(y_up_candidate)
    return y_up


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print
       # "---  new folder...  ---"
        print
       # "---  OK  ---"

    else:
        print
       # "---  There is this folder!  ---"

def parseXmlFiles(ann_file, img_prefix, patch_path):
    ps = 1
    num_pic = 0
    img_ids = list_from_file(ann_file)
    for delta_angle in np.arange(0,360,0.1):
        for img_id in img_ids:

            xml_file = os.path.join(img_prefix, 'annotations_SSDD++',
                                f'{img_id}.xml')
            img_name = os.path.join(img_prefix, 'JPEGImages',
                                f'{img_id}.jpg')

            num_pic += 1
            num = 0

            tree = ET.parse(xml_file)
            root = tree.getroot()
            if root.tag != 'annotation':
                raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

            labels = []
            labels_scaled = []

            for elem in root:  # root.tag = Annotation
                if elem.tag == "filename":  # get picture name
                    pic_name = elem.text
                    #if ps: print(pic_name)
                    # img_name = os.path.join(image_path, pic_name + '.jpg')

                    img = cv2.imdecode(np.fromfile(u'{}'.format(img_name), dtype=np.uint8), 1)
                    # 转换为PIL.IMAGE格式
                    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                    # width=int(image.size[0])
                    # height=int(image.size[1])

                if elem.tag == "object":  # elem.tag = frame,object
                    is_append = True
                    is_end = False
                    for subelem in elem:  # subelem.tag = name,polygon

                        if is_append:  # if list was just appended, reintialize the polygon
                            polygon = dict()
                            polygon_scaled = dict()
                            is_append = False
                        if subelem.tag == "name":
                            polygon["name"] = polygon_scaled["name"] = subelem.text
                        if subelem.tag == "polygon":  # option.tag =
                            for option in subelem:
                                if option.tag == 'x1':
                                    polygon["x1"] = int(option.text)
                                if option.tag == 'y1':
                                    polygon["y1"] = int(option.text)
                                if option.tag == 'x2':
                                    polygon["x2"] = int(option.text)
                                if option.tag == 'y2':
                                    polygon["y2"] = int(option.text)
                                if option.tag == 'x3':
                                    polygon["x3"] = int(option.text)
                                if option.tag == 'y3':
                                    polygon["y3"] = int(option.text)
                                if option.tag == 'x4':
                                    polygon["x4"] = int(option.text)
                                if option.tag == 'y4':
                                    polygon["y4"] = int(option.text)

                                    is_end = True
                        if is_end:  # if all location and class of current polygon have been read, append current polygon to list pool
                            num += 1
                            is_end = False
                            is_append = True
                            #old = (polygon["x1"],polygon["y1"],polygon["x2"],polygon["y2"])  # raw manual bounding box
                            # if ps:  #print("old polygon", old)
                            # polygon_scaled["x1"], polygon_scaled["y1"], polygon_scaled["x2"], polygon_scaled["y2"] = \
                            #     scale_bounding_box(old, (width,height), 1.3)  # scale the bounding box
                            # if ps: #print("scaled polygon:", polygon_scaled["x1"], polygon_scaled["y1"],
                                        # polygon_scaled["x2"], polygon_scaled["y2"])

                            labels.append(polygon)  # add raw bounding box to labels
                            labels_scaled.append(polygon_scaled)  # add scaled bounding box to labels

                            # show every bounding box on current image successively
                            #draw_caption(img, old,"{}th pic: ".format(num_pic)+ pic_name+": {}th Target".format(num))
                            x1,x2,x3,x4,y1,y2,y3,y4= polygon["x1"],polygon["x2"],polygon["x3"],polygon["x4"],polygon["y1"],polygon["y2"],polygon["y3"],polygon["y4"]
                            x12_mean = (x1 + x2) / 2
                            y12_mean = (y1 + y2) / 2
                            x34_mean = (x3 + x4) / 2
                            y34_mean = (y3 + y4) / 2
                            x_center = (x12_mean + x34_mean) / 2
                            y_center = (y12_mean + y34_mean) / 2
                            xmin = min(x1, x2, x3, x4)
                            ymin = min(y1, y2, y3, y4)
                            xmax = max(x1, x2, x3, x4)
                            ymax = max(y1, y2, y3, y4)
                            square_length=max(xmax-xmin,ymax-ymin)
                            eps=0.000000001
                            #image.show()
                            region_raw=image.crop([x_center-square_length/2,y_center-square_length/2,x_center+square_length/2,y_center+square_length/2])
                            #region_raw.show()
                            mkdir(patch_path + "target/Correlation/raw")
                            region_raw.save(patch_path + "target/Correlation/raw/" + pic_name + "{}th_tgt.jpg".format(num))
                            slope=(y34_mean-y12_mean)/(x34_mean-x12_mean+eps)
                            angle=math.atan(slope)*180/math.pi
                            image_center_rotated = image.rotate(angle-delta_angle,center=(x_center,y_center))
                            # image_center_rotated.show()
                            region_rotated=image_center_rotated.crop([x_center - square_length / 2, y_center - square_length / 2, x_center + square_length / 2,y_center + square_length / 2])
                            #region_rotated.show()
                            mkdir(patch_path + "target/Correlation/{:0>5.1f}".format(delta_angle))
                            region_rotated.save(patch_path + "target/Correlation/{:0>5.1f}/".format(delta_angle) + pic_name + "{}th_tgt.jpg".format(num))

                            # region.save(patch_path+"target/" + pic_name+"{}th_tgt.jpg".format(num))
if __name__ == '__main__':
    if 1:
        ann_file = "SSDD+/ImageSets/Main/train.txt"
        img_prefix = "SSDD+/"
        patch_path = "SSDD+/rotation/train/"
        parseXmlFiles(ann_file, img_prefix, patch_path)
    if 1:
        ann_file = "SSDD+/ImageSets/Main/val.txt"
        img_prefix = "SSDD+/"
        patch_path = "SSDD+/rotation/val/"
        parseXmlFiles(ann_file, img_prefix, patch_path)
    if 1:
        ann_file = "SSDD+/ImageSets/Main/trainval.txt"
        img_prefix = "SSDD+/"
        patch_path = "SSDD+/rotation/trainval/"
        parseXmlFiles(ann_file, img_prefix, patch_path)


