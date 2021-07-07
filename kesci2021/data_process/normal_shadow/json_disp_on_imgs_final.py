import json
import os
import argparse
import cv2
import numpy as npmshow
# underwater_classes = ['holothurian', 'echinus', 'scallop', 'starfish']
underwater_classes = ['cube', 'ball', 'cylinder', 'human body', 'tyre', 'square cage', 'circle cage', 'metal bucket']
def parse_args():
    parser = argparse.ArgumentParser(description='json2submit_nms')
    parser.add_argument('--test_json', help='test result json', type=str)       #####   2nd
    parser.add_argument('--submit_file', help='submit_file_name', type=str)      #######   3rd
    args = parser.parse_args()
    return args


def my_iou(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: IOU.
    """

    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)
        # return S_cross/S2

def wh2xy(rec_wh):
    x0_,y0_,w,h= rec_wh
    x0=x0_
    y0=y0_
    x1=x0_+w
    y1=y0_+h
    return [x0,y0,x1,y1]



if __name__ == '__main__':
    args = parse_args()
    # img_path = '/home/p/HD1/深度学习竞赛/sonardet2021/data/train/image_he/'
    img_path = '/media/n/SanDiskSSD/HardDisk/data/kesci2021/acoustics/test_A/VOC2007/JPEGImages/'
    # detection_with_shadow = '/home/p/HD1/深度学习竞赛/sonardet2021/work_dirs/cascade_rcnn_r50_fpn_with_shad_right/bs2_lr_real_0025/epoch_12.bbox.json'
    # detection_with_shadow= '/home/n/Github/Myrepo/LearningNotes/kesci2021/data_process/normal_shadow/epoch_10_result.bbox.json'
    # test_json_with_shadow = json.load(open("/home/n/Github/Myrepo/LearningNotes/kesci2021/data_process/normal_shadow/annotations_box_with_shadow_right/testA.json", "r"))     ###### 1st

    detection_normal = '/home/n/Github/Myrepo/LearningNotes/kesci2021/data_process/normal_shadow/output/epoch_12.bbox.json'
    test_json_normal = json.load(open("/home/n/Github/Myrepo/LearningNotes/kesci2021/data_process/normal_shadow/annotations_box_with_shadow_right/testA.json", "r"))     ###### 1st

    # test_json = json.load(open(detection_with_shadow, "r"))
    normal_json = json.load(open(detection_normal, "r"))
    submit_file_name = args.submit_file
    submit_path = 'submit/'
    os.makedirs(submit_path, exist_ok=True)
    img = test_json_normal['images']
    images = []

    # img_id2name = {}
    # with open(detection_normal) as fr:
    #     coco_test_json_file = json.load(fr)
    # imgs_list = coco_test_json_file['images']
    # """切分测试图像时产生的json  croped_testA.json
    # {"file_name": "0_1.bmp", "height": 512, "width": 512, "id": 1, "yshift": 0}, {"file_name": "0_2.bmp", "height": 512, "width":
    # 512, "id": 2, "yshift": 460}, {"file_name": "0_3.bmp", "height": 512, "width": 512, "id": 3, "yshift": 920},
    # {"file_name": "0_4.bmp", "height": 460, "width": 512, "id": 4, "yshift": 1380},
    # """
    # for imgs_ins in imgs_list:
    #     # img_id2name[imgs_ins['id']] = [imgs_ins['file_name'], (imgs_ins['width'], imgs_ins['height']), imgs_ins['Ymin']]
    #     img_id2name[imgs_ins['file_name']] = imgs_ins['id']



    # csv_file = open(submit_file_name, 'w')
    csv_file=open('output/epoch_12.csv','w')
    csv_file.write("name,image_id,confidence,xmin,ymin,xmax,ymax\n")
    imgid2anno = {}
    imgid2n_anno = {}
    imgid2name = {}
    n_name2_imgid = {}
    for imageinfo in test_json_normal['images']:
        imgid = imageinfo['id']
        imgid2name[imgid] = imageinfo['file_name']
    # for anno in test_json:
    #     img_id = anno['image_id']
    #     if img_id not in imgid2anno:
    #         imgid2anno[img_id] = []
    #     imgid2anno[img_id].append(anno)


    for n_imageinfo in test_json_normal['images']:
        imgid = n_imageinfo['id']
        imgfilename = n_imageinfo['file_name']
        n_name2_imgid[imgfilename] = imgid

    for n_anno in normal_json:
        n_img_id = n_anno['image_id']
        if n_img_id not in imgid2n_anno:
            imgid2n_anno[n_img_id] = []
        imgid2n_anno[n_img_id].append(n_anno)


    n_w = 2
    score_threshold_with_shadow = 0.6
    score_threshold_cascade = 0

    for imgid in imgid2n_anno:   # normal
        annon = imgid2n_anno[imgid]
        bbox_old= {'bbox':[0,0,0,0]}
        for n in annon:
            n['bbox']=wh2xy(n['bbox'])
            iou=my_iou(bbox_old['bbox'], n['bbox'])
            if iou > 0.9 :
                score_index=np.argmax((n['score'],bbox_old['score']))
                if score_index==1:
                    n['category_id']=bbox_old['category_id']
                else:
                    bbox_old['category_id']= n['category_id']
            bbox_old =  n
        for n in annon:
            confidence = n['score']

            class_id = int(n['category_id'])
            class_name = underwater_classes[class_id-1]

            image_name = imgid2name[imgid]
            image_id = image_name.split('.')[0]    # + '.xml'

            csv_file.write(class_name + ',' + image_id + ',' + str(confidence) + ',' + str(n['bbox'][0])+ ',' + str(n['bbox'][1]) + ',' + str(n['bbox'][2]) + ',' + str(n['bbox'][3]) + '\n')



        img_name_tmp = imgid2name[imgid]
        # print("    img_name_tmp : ",img_name_tmp)
        src = cv2.imread(img_path + img_name_tmp)
        # cv2.namedWindow("comparison")

        nn_img_id = n_name2_imgid[img_name_tmp]
        n_annos = imgid2n_anno[nn_img_id]
        for anno in n_annos:  # normal
            # xmin, ymin, w, h = anno['bbox']
            xmin, ymin, xmax, ymax = anno['bbox']
            # xmax = xmin + w
            # ymax = ymin + h
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            confidence = anno['score']
            class_id = int(anno['category_id'])
            class_name = underwater_classes[class_id-1]
            image_name = imgid2name[imgid]
            image_id = image_name.split('.')[0]    # + '.xml'
            # csv_file.write(class_name + ',' + image_id + ',' + str(confidence) + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + '\n')

            if confidence >= score_threshold_cascade:
                cv2.rectangle(src, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), n_w)
                cv2.putText(src, '{}|{:.3f}'.format(class_name, confidence),
                            (int(xmin - 24), int(ymax)+90),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), n_w)

        # cv2.imshow("comparison", src)
        # # cv2.imwrite('visual_results/' + img_name, src)
        # cv2.waitKey(0)
        #

    # if len()
    #     csv_file.write(class_name + ',' + image_id + ',' + str(confidence) + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + '\n')
    csv_file.close()