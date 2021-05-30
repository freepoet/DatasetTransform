import json

if __name__ == '__main__':
    img_path = '/media/n/SanDiskSSD/HardDisk/data/kesci2021/acoustics/test_A/VOC2007/JPEGImages/'
    # detection_with_shadow = '/home/p/HD1/深度学习竞赛/sonardet2021/work_dirs/cascade_rcnn_r50_fpn_with_shad_right/bs2_lr_real_0025/epoch_12.bbox.json'
    detection_with_shadow = '/home/n/Github/Myrepo/LearningNotes/kesci2021/data_process/normal_shadow/epoch_10_result.bbox.json'
    test_json_with_shadow = json.load(open(
        "/home/n/Github/Myrepo/LearningNotes/kesci2021/data_process/normal_shadow/annotations_box_with_shadow_right/testA.json",
        "r"))  ###### 1st

    detection_normal = '/home/n/Github/Myrepo/LearningNotes/kesci2021/data_process/normal_shadow/epoch_12.bbox.json'
    test_json_normal = json.load(open(
        "/home/n/Github/Myrepo/LearningNotes/kesci2021/data_process/normal_shadow/annotations_box_with_shadow_right/testA.json",
        "r"))  ###### 1st
