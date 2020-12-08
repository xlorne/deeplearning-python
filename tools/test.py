#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File Name: test.py
# Description:
# Create Time: 2020-12-07 09:05
# Author: lorne



import os
import shutil
import glob

# 遍历文件夹
def walkFile(file):
    for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        for f in files:
            if os.path.splitext(f)[-1] == ".json":
                print(os.path.join(root, f))
                fileName = os.path.splitext(f)[0]

                # if fileName.find("(")!=-1:
                #     print(fileName)

                jsonNewFile = os.path.join("/mnt/images/",fileName+".json")
                jsonOldFile = os.path.join(root, fileName+".json")
                #
                jpgNewFile = os.path.join("/mnt/images/", fileName + ".jpg")
                jpgOldFile = os.path.join(root, fileName + ".jpg")

                if os.path.exists(jpgOldFile):
                    shutil.copyfile(jsonOldFile,jsonNewFile)
                    shutil.copyfile(jpgOldFile, jpgNewFile)

                # print(jsonNewFile)



        # 遍历所有的文件夹
        for d in dirs:
            print(os.path.join(root, d))


def main():
    walkFile("/mnt/缺陷原始照片/销钉新")


if __name__ == '__main__':
    main()
    print("over")