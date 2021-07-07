# 用于显示前视声呐和前视声呐图像的框
import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np
import shutil

def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def parseXmlFiles(xml_path, image_path, new_path_for_save_wrong_xml):
    num_pic = 0
    file_dict = {id: name for id, name in enumerate(os.listdir(xml_path))}
    len_file_dict = len(file_dict)
    num_pic_st_len = True
    while num_pic_st_len:
        if num_pic >= len_file_dict - 1:
            num_pic_st_len = False
        f = file_dict[num_pic]

        num = 0
        # print(f)
        if not f.endswith('.xml'):  # jump off non-xml files
            continue
        labels = []
        xml_file = os.path.join(xml_path, f)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        for elem in root:  # root.tag = Annotation
            if elem.tag == "filename":
                pic_name = elem.text[:-4]
                img_name = os.path.join(image_path, pic_name + '.bmp')
                # print(img_name)
                # img = cv2.imread(img_name, cv2.IMREAD_COLOR)
                img = cv2.imdecode(np.fromfile(u'{}'.format(img_name), dtype=np.uint8), 1)
            if elem.tag == "object":  # elem.tag = frame,object
                is_append = True
                is_end = False
                for subelem in elem:  # subelem.tag = name,bndbox

                    if is_append:  # if list was just appended, reintialize the bndbox
                        bndbox = dict()
                        is_append = False
                    if subelem.tag == "name":
                        bndbox["name"] = subelem.text
                    if subelem.tag == "bndbox":  # option.tag = xmin,ymin,xmax,ymax
                        for option in subelem:
                            if option.tag == 'xmin':
                                x1 = int(option.text)
                            if option.tag == 'ymin':
                                y1 = int(option.text)
                            if option.tag == 'xmax':
                                x2 = int(option.text)
                            if option.tag == 'ymax':
                                y2 = int(option.text)
                                is_end = True
                    if is_end:  # if all location and class of current bndbox have been read, append current bndbox to list pool
                        num += 1
                        labels.append(bndbox)
                        is_end = False
                        is_append = True

                        draw_caption(img, (x1, y1, x2, y2), bndbox["name"]+":"+pic_name)
                        # print(img)
                        # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                        # cv2.namedWindow("img", 1)
                        xy = (0, 0)

                        def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
                            if event == cv2.EVENT_LBUTTONDOWN:
                                # xy = "%d,%d" % (x, y)
                                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                                cv2.rectangle(img, (x1, y), (x2, y2), color=(255, 0, 0), thickness=2)
                                # cv2.imshow('img', img)
                        # cv2.imshow('img', img)
                        # cv2.setMouseCallback("img", on_EVENT_LBUTTONDOWN)
                        # print(xy)

                        # xy = (0, 0)
                        # cv2.imshow('img', img)
                        # cv2.waitKey(0)
        # display new annotated img
        cv2.imshow('Press Enter to pass, or space to select it as wrong: ', img)
        #

        waitkey_num = cv2.waitKeyEx()
        if waitkey_num == 32:  # space
            n_tmp = input("Input 'jump' for jump back, or input anything else for select is as wrong:")
            if n_tmp == 'jump':
                print('Current pic number is : ', num_pic)
                n_tmp = input("Enter the numbers you need to jump:")
                num_pic = int(n_tmp)
            else:
                shutil(xml_file, os.path.join(new_path_for_save_wrong_xml, f))
            cv2.destroyAllWindows()
        if waitkey_num == 13:  # enter
            num_pic += 1
            cv2.destroyAllWindows()
            # save_xml_to_new_path()





if __name__ == '__main__':
    xml_path = '/media/n/SanDiskSSD/HardDisk/data/kesci2021/acoustics/train/训练集/box/'
    new_path_for_save_wrong_xml = '/media/n/SanDiskSSD/HardDisk/data/kesci2021/acoustics/train/训练集/box_wrong/'
    #image_path = "./image_he/"
    image_path = '/media/n/SanDiskSSD/HardDisk/data/kesci2021/acoustics/train/V训练集/image_he/'
    #xml_path = './train/侧视声呐图像/box/'
    #image_path = "./train/侧视声呐图像/image/"
    # os.mkdirs(new_path_for_save_wrong_xml)
    parseXmlFiles(xml_path, image_path, new_path_for_save_wrong_xml)
