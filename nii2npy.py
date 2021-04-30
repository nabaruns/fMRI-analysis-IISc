#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
import multiprocessing.pool
import numpy as np
import os
from nilearn import image as nimage


from os import listdir
from os.path import isfile, join
mypath="./"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for p in onlyfiles:
    img = nimage.load_img(os.path.join(mypath,p)).get_data()
    print(p)
    np.save(p.split(".")[0], img)
