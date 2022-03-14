# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 01:13:10 2020

@author: Phoenix
"""
import os
import shutil
import requests

def get_from_url(urlpath, auth_string=None):
    '''Get file from urlpath'''
    headers = auth_string
    req = requests.get(urlpath, headers=headers)
    local_filename = urlpath.split('/')[-1]
    with open(local_filename, 'wb') as file:
        file.write(req.content)
        print(f'{local_filename} saved at {os.getcwd()}')
        
    return

if __name__=='__main__':
    filepath = r'https://github.com/ycx91/ycx-downloads/raw/master/aiap6week8projectdownloads/checkpoints.7z'
    get_from_url(filepath)