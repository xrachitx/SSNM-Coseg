import torch
from torchvision import transforms
import csv
import queue
import copy
import random
import os
import numpy as np
from PIL import Image, ImageOps
import cv2


def filt_small_instance(coco_item, pixthreshold=4000,imgNthreshold=5):
    list_dict = coco_item.catToImgs
    for catid in list_dict:
        list_dict[catid] = list(set( list_dict[catid] ))
    new_dict = copy.deepcopy(list_dict)
    for catid in list_dict:
        imgids = list_dict[catid]
        for n in range(len(imgids)):
            imgid = imgids[n]
            anns = coco_item.imgToAnns[imgid]
            has_large_instance = False
            for ann in anns:
                if (ann['category_id'] == catid) and (ann['iscrowd'] == 0) and (ann['area'] > pixthreshold):
                    has_large_instance = True
            if has_large_instance is False:
                new_dict[catid].remove(imgid)
        imgN = len(new_dict[catid])
        if imgN <imgNthreshold:
            new_dict.pop(catid)
            print('catid:%d  remain %d images, delet it!'%(catid,imgN))
        else:
            print('catid:%d  remain %d images' % (catid, imgN))
    print('remain  %d  categories'%len(new_dict))
    np.save('./utils/new_cat2imgid_dict%d.npy'%pixthreshold, new_dict)
    return new_dict

def co_skel_data_producer(cat2imgpath,q,batch_size=5, group_size=5, max_images=50, img_size=224,gt=0, cls_size=13):
    img_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    gt_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    img_transform_gray = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),transforms.Normalize(mean=[0.449], std=[0.226])])

    cat2index = {  
       "Aeroplane": 1,
       "Bear": 2,
       "Bird": 3,
       "Bus": 4,
       "Cats": 5,
       "Cow": 6,
       "Cycle": 7,
       "Dog": 8,
       "Elephant": 9,
       "Giraffe":10,
       "Horse":11,
       "Sheep":12,
       "Zebra":13
    }
    
    while q.qsize()<40:
        print(q.qsize())
        rgb = torch.zeros(batch_size*group_size, 3, img_size, img_size)
        cls_labels = torch.zeros(batch_size, cls_size)
        mask_labels = torch.zeros(batch_size*group_size, img_size, img_size)

        sel_cats = random.sample(cat2imgpath.keys(), min(len(cat2imgpath.keys()), batch_size))

        img_n = 0
        group_n = 0
        for cat in sel_cats:
            random.shuffle(cat2imgpath[cat])
            for i in range(group_size):
                img_path = cat2imgpath[cat][i][0]

                img = Image.open(img_path)
                if img.mode == 'RGB':
                    img = img_transform(img)
                else:
                    img = img_transform_gray(img)

                mask_path = cat2imgpath[cat][i][1]
                # mask = Image.open(mask_path)
                mask = cv2.imread(mask_path)
                mask = mask*255
                # mask = ImageOps.grayscale(mask)
                mask[mask > 0] = 255
                mask = gt_transform(mask)
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                rgb[img_n,:,:,:] = copy.deepcopy(img)
                mask_labels[img_n,:,:] = copy.deepcopy(mask)
                img_n += 1

            cls_labels[group_n, cat2index[cat]-1] = 1
            group_n += 1

        q.put([rgb, cls_labels, mask_labels])
    print("yay")
if __name__ == "__main__":
    q = co_skel_data_producer("./final.csv")
    print(q.get())
    print("yay")