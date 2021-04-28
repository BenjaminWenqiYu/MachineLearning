#!/usr/bin/env python3.7
# -*-coding:utf-8 -*-
# @Time  : 2020-10-28 11:16
# Author : yuwenqi
import numpy as np
a = np.arange(6).reshape(2, 3) + 10
print(a)
print(np.argmax(a, axis=0))
print(np.argmax(a, axis=1))