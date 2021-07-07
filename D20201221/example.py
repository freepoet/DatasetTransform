# -*- coding: utf-8 -*-
"""
@File    : example.py
@Time    : 12/22/20 3:08 PM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
12/22/20 3:08 PM      1.0         None
# @Software: PyCharm
"""
import fire
def add(x,y):
    return x+y

def mul(**kwargs):
    a=kwargs['a']
    b=kwargs['b']
    return a*b

if __name__=='__main__':
    fire.Fire()