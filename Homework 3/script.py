from __future__ import print_function
import yaml
import sys
import os
from PIL import Image
import xml.etree.ElementTree as ET
import shutil
import pickle


dictionary = {}


def xmlparsing(path):
    global dictionary
    files = os.listdir(path)
    print("Parsing the XML Files ...")
    for file in files:
        fullname = os.path.join(path, file)
        tree = ET.parse(fullname)
        root = tree.getroot()
        list_obj = []
        file_name = root.find("filename").text.replace('.jpg', '')
        # print(file_name)
        for child in root:
            # print(child.tag)
            if child.tag == "object":
                for step_child in child:
                    # print("***")
                    if step_child.tag == "name":
                        name = step_child.text
                    if step_child.tag == "bndbox":
                        for step_step_child in step_child:
                            if step_step_child.tag == "xmin":
                                x_min = float(step_step_child.text)
                            if step_step_child.tag == "ymin":
                                y_min = float(step_step_child.text)
                            if step_step_child.tag == "xmax":
                                x_max = float(step_step_child.text)
                            if step_step_child.tag == "ymax":
                                y_max = float(step_step_child.text)
                        img_obj = Image_Object(name, x_min, y_min, x_max, y_max)
                        if file_name not in dictionary:
                            dictionary[file_name] = {}
                            dictionary[file_name][name] = []
                            dictionary[file_name][name].append(img_obj)
                        else:
                            if name not in dictionary[file_name]:
                                dictionary[file_name][name] = []
                                dictionary[file_name][name].append(img_obj)
                            else:
                                dictionary[file_name][name].append(img_obj)
    print(next(iter(dictionary)))
    print("XML Files Parsing done")
    # exit(0)



def binaryconversion(data):
    binary_path = "D:/Masters/Deep Learing/Homework 3/VOCtrainval_11-May-2012/BinaryOutput/"
    if not os.path.exists(binary_path):
        os.makedirs(binary_path)
    full_path = os.path.join(binary_path, "Assignment.bin")
    output_file = open(full_path, "wb")
    for x in data:
        pickle.dump(x, output_file)
    output_file.close()



class Image_Object():
    name = ""
    bndbox_x_min = 0.0
    bndbox_y_min = 0.0
    bndbox_x_max = 0.0
    bndbox_y_max = 0.0

    def __init__(self, name, x_min, y_min, x_max, y_max):
        self.name = name
        self.bndbox_x_min = x_min
        self.bndbox_y_min = y_min
        self.bndbox_x_max = x_max
        self.bndbox_y_max = y_max



def parsing_into_folder(path, type):
    global dictionary
    print("Parsing Images ...")
    maxsize = 244, 244
    i = 0
    list_images = []
    j=0
    for key, value in dictionary.items():
        im = Image.open(path + key + ".jpg")
        j=j+1
        # print(im.format, im.size, im.mode)
        list = value
        for key1, value1 in value.items():
            list = value1
            for x in list:
                i = i + 1
                x_min = x.bndbox_x_min
                y_min = x.bndbox_y_min
                x_max = x.bndbox_x_max
                y_max = x.bndbox_y_max
                box = (x_min, y_min, x_max, y_max)
                region = im.crop(box)
                # region.thumbnail(maxsize, Image.ANTIALIAS)
                region = region.resize(maxsize, Image.ANTIALIAS)
                if type == "HardDisk":
                    file_name = "".join([x.name, "_", str(i), ".JPEG"])
                    output_path = "D:/Masters/Deep Learing/Homework 3/VOCtrainval_11-May-2012/Output/" + key1
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    o_path = os.path.join(output_path, file_name)
                    region.save(o_path, quality = 95)
                    list_images.append(region)
                elif type == "InMemory":
                    list_images.append(region)
                    # binaryconversion(list_images)
        binaryconversion(list_images)
            # exit()


def main():

    if not os.path.exists("D:/Masters/Deep Learing/Homework 3/VOCtrainval_11-May-2012/Output"):
        os.makedirs("D:/Masters/Deep Learing/Homework 3/VOCtrainval_11-May-2012/Output")
    img_classes_path = "D:/Masters/Deep Learing/Homework 3/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/"
    xml_files_path = "D:/Masters/Deep Learing/Homework 3/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/"
    img_files_path = "D:/Masters/Deep Learing/Homework 3/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/"
    global dictionary
    input=raw_input("Enter HardDisk or InMemory\n")

    xmlparsing(xml_files_path)
    if input == "HardDisk":
        # exit()
        parsing_into_folder(img_files_path, input)
        print("Finished")
    elif input == "InMemory":
        parsing_into_folder(img_files_path, input)


if __name__ == "__main__":
    sys.exit(main())
