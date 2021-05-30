# 用于显示前视声呐和前视声呐图像的框
import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np
import shutil

headstr = """\
<annotation>
    <sonar>
        <type>Tritech 1200ik</type>
        <version>1</version>
        <range>10.0014</range>
        <horiangle>120</horiangle>
        <vertiangle>12</vertiangle>
        <soundspeed>1466.2</soundspeed>
        <frequency>1200k</frequency>
    </sonar>
    <folder>VOC</folder>
    <filename>%s</filename>
"""
objstr = """\
    <object>
        <name>%s</name>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def parseXmlFiles(xml_path, image_path, xml_path_for_box_with_shadow_tmp, xml_path_for_box_with_shadow_final, xml_path_for_wrong_box):
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
        print("#######---------########---------A New Image for you to Check its Annotation-------#######----------")

        xml_file_for_box_with_shadow = os.path.join(xml_path_for_box_with_shadow_tmp, f)
        xml_file_for_box_with_shadow_final = os.path.join(xml_path_for_box_with_shadow_final, f)
        xml_file_for_box_wrong = os.path.join(xml_path_for_wrong_box, f)
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))
        f_xml_with_shadow = open(xml_file_for_box_with_shadow, "w")
        head = headstr % (f)
        f_xml_with_shadow.write(head)


        for elem in root:  # root.tag = Annotation
            if elem.tag == "filename":
                pic_name = elem.text[:-4]
                img_name = os.path.join(image_path, pic_name + '.bmp')
                print("current pic path is : ", img_name)
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
                        f_xml_with_shadow.write(objstr % (bndbox["name"], x1, y1, x2, y2))
                        # labels.append(bndbox)
                        is_end = False
                        is_append = True

                        draw_caption(img, (x1, y1, x2, y2), bndbox["name"]+":"+pic_name)
                        # print(img)
                        # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                        cv2.namedWindow("img", 1)
                        xy = (0, 0)

                        def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
                            if event == cv2.EVENT_LBUTTONDOWN:
                                # xy = "%d,%d" % (x, y)
                                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                                cv2.rectangle(img, (x1, y), (x2, y2), color=(255, 0, 0), thickness=2)
                                f_xml_with_shadow.write(objstr % (bndbox["name"]+' with shad', x1, y, x2, y2))
                                cv2.imshow('img', img)
                        cv2.imshow('img', img)
                        cv2.setMouseCallback("img", on_EVENT_LBUTTONDOWN)
                        # print(xy)

                        xy = (0, 0)
                        cv2.imshow('img', img)
                        cv2.waitKey(0)
        # display new annotated img
        f_xml_with_shadow.write(tailstr)
        f_xml_with_shadow.close()
        print(num_pic,"th xml: ",xml_file_for_box_with_shadow, ' is processed')
        cv2.imshow('Press Enter to save xml_with_shad, or space to edit: ', img)
        #

        waitkey_num = cv2.waitKeyEx()
        if waitkey_num == 32:
            n_tmp = input("Input 'wrong' for select it as wrong annotated, 'delete' for redo, or  anything else for jump back:")
            if n_tmp == 'wrong':
                shutil.copyfile(xml_file, xml_file_for_box_wrong)
                print(num_pic, "th xml: ", xml_file, ' is moved to wrong path')
                num_pic += 1
            elif  n_tmp == 'delete':
                print(num_pic, "th xml: ", xml_file, ' is deleted, you need to reannotated now')
                os.remove(xml_file_for_box_with_shadow)
                # num_pic = num_pic
            else:
                print('Current pic number is : ', num_pic)
                n_tmp = input("Enter the numbers you need to jump:")
                num_pic = int(n_tmp)
            cv2.destroyAllWindows()
        if waitkey_num == 13:
            shutil.copyfile(xml_file_for_box_with_shadow, xml_file_for_box_with_shadow_final)
            print(num_pic, "th xml: ", xml_file_for_box_with_shadow, ' is annotated and moved to right path')
            num_pic += 1
            cv2.destroyAllWindows()
            # save_xml_to_new_path()





if __name__ == '__main__':
    xml_path = '/media/n/SanDiskSSD/HardDisk/data/kesci2021/acoustics/train/训练集/box_with_shadow/'
    xml_path_for_box_with_shadow_tmp = '/media/n/SanDiskSSD/HardDisk/data/kesci2021/acoustics/train/训练集/box_with_shadow_tmp/'
    xml_path_for_box_with_shadow ='/media/n/SanDiskSSD/HardDisk/data/kesci2021/acoustics/train/训练集/box_with_shadow/'
    xml_path_for_wrong_box = '/media/n/SanDiskSSD/HardDisk/data/kesci2021/acoustics/train/训练集/box_wrong/'
    #image_path = "./image_he/"
    image_path = '/media/n/SanDiskSSD/HardDisk/data/kesci2021/acoustics/train/训练集/image_he/'
    #xml_path = './train/侧视声呐图像/box/'
    #image_path = "./train/侧视声呐图像/image/"

    parseXmlFiles(xml_path, image_path, xml_path_for_box_with_shadow_tmp, xml_path_for_box_with_shadow, xml_path_for_wrong_box)
