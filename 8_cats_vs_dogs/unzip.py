# -*- coding: utf-8 -*-
"""
Created on Thu May 30 01:12:16 2019

@author: XARC
Unzips the zip file to a folder
"""
import zipfile
import sys

def unzip(SOURCE, DEST):
    if DEST=='':
        DEST = SOURCE + '//..'
    zipref = zipfile.ZipFile(SOURCE, 'r')
    zipref.extractall(DEST)
    zipref.close()

if __name__=='__main__':
    try:
        dest = sys.argv[2]
    except:
        dest = ''
    unzip(sys.argv[1], dest)

