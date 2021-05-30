import json
import datetime
import numpy as np
import re

class JsonEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)

def load_dict(filename):
    '''load dict from json file'''
    with open(filename, "r") as json_file:
        dic = json.load(json_file)
    return dic


def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename, 'w') as json_file:
        json.dump(dic, json_file, ensure_ascii=False, cls=JsonEncoder)

# a={'a':1,'b':2,'c':3}
# for k,v in a.items():
#     print(k)
#     print(v)
# # save_dict('./dict',a)
#
# b=load_dict('./dict')
# str='calibration=0.9863265752792358 tensorflow:Final best valid   0 loss=0.20478513836860657 pr=0.39401692152023315 rate=0.提取  '
# str2='imagewidth=1024 log_2021-03-15-100239_annotaion_48.xmlhuman body_1.jpg'
# # 匹配“calibration=”后面的数字
# pattern = re.compile(r'(?<=imagewidth=)\d+\.?\d*')
# a=pattern.findall(str2)
