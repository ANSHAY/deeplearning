# -*- coding: utf-8 -*-
"""
Created on Thu May 30 00:23:55 2019

@author: XARC
Partitions the dataset

Partitions the dataset from SOURCE into DESTINATION directory
containing train and test directories which contain data for
different CLASSES in their respective directories split as per
the SPLIT parameter.
It defines CLASSES based on the directories present in SOURCE directory
Note: SOURCE directory should not contain any file
ARGUMENTS: SOURCE, DESTINATION, SPLIT
RETURNS: 
"""
import os
import sys
import random
from shutil import copyfile

def split_data(SOURCE, DESTINATION, SPLIT):
    # create destination diectories
    CLASSES = os.listdir(SOURCE)
    print("\nClasses are: ")
    print(CLASSES)
    try:
        os.mkdir(DESTINATION, mode=0o777)
        os.mkdir(DESTINATION + 'train', mode=0o777)
        os.mkdir(DESTINATION + 'test', mode=0o777)
        os.mkdir(DESTINATION + 'val', mode=0o777)
        for c in CLASSES:
            os.mkdir(DESTINATION + 'train' + os.sep + c, mode=0o777)
            os.mkdir(DESTINATION + 'val' + os.sep + c, mode=0o777)
            os.mkdir(DESTINATION + 'test' + os.sep + c, mode=0o777)
    except:
        print("\nCould not create directories.")
        pass
    # Split the data
    for c in CLASSES:
        list_dir = os.listdir(SOURCE + c)
        dir_len = len(list_dir)
        SPLIT_INDEX_TRAIN =  dir_len * SPLIT
        SPLIT_INDEX_VAL = dir_len * (1 + SPLIT) // 2
        list_dir = random.sample(list_dir, dir_len)
        print('\nCopying files for ' + c + '\n')
        for i in range(dir_len):
            if (i % (dir_len/10) == 0):
                print('.')
            if (i < SPLIT_INDEX_TRAIN):
                dest = 'train'
            elif (i < SPLIT_INDEX_VAL):
                dest = 'val'
            else:
                dest = 'test'
            if (os.path.getsize(SOURCE+c+os.sep+list_dir[i]) > 0):
                copyfile(SOURCE+c+os.sep+list_dir[i],
                         DESTINATION+dest+os.sep+c+os.sep+list_dir[i])
            else:
                print ('Skipping {0} because of empty file'.format(list_dir[i]))

if __name__ == '__main__':
    try:
        source = sys.argv[1]
        dest = sys.argv[2]
        split = float(sys.argv[3])
    except:
        print ("\nFeed correct arguments")
    split_data(source, dest, split)
