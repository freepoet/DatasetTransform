# -*- coding: utf-8 -*-
"""
@File    : test616.py
@Time    : 12/22/20 8:36 AM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/22/20 8:36 AM      1.0         None
# @Software: PyCharm
"""
try:
    import ipdb
except:
    import pdb as ipdb

def sum(x):
    r=0
    for ii in x:
        r+=ii
    return r
def mul(x):
    r=1
    for ii in x:
        r*=ii
    return r
ipdb.set_trace()
x=[1,2,3,4,5]
y=[2,2,2,2,2]
r1=sum(x)
r2=mul(x)
r3=sum(y)
r4=mul(y)
