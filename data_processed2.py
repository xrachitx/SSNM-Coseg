import torch
from torchvision import transforms
import csv
import queue
import copy
import random
import os
import numpy as np
from PIL import Image, ImageOps


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

def co_skel_data_producer(csv_file,batch_size=10, group_size=5, max_images=20, img_size=224,gt=0, cls_size=13):
    img_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    gt_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
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

    cat2imgpath = {  
       "Aeroplane":[],
       "Bear":[],
       "Bird":[],
       "Bus":[],
       "Cats":[],
       "Cow":[],
       "Cycle":[],
       "Dog":[],
       "Elephant":[],
       "Giraffe":[],
       "Horse":[],
       "Sheep":[],
       "Zebra":[]
    }
    csv_rows = []
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            csv_rows.append(row)

    random.shuffle(csv_rows)

    for row in csv_rows:
        if len(cat2imgpath[row[-1]]) < max_images:
            if gt == 0:
                cat2imgpath[row[-1]].append([row[0], row[3]])
            elif gt == 1:#Complete Seg Mask
                cat2imgpath[row[-1]].append([row[0], row[2]])
            else:#Actual Seg Masks
                cat2imgpath[row[-1]].append([row[0], row[1]])
        else:
            pass
    

    q = queue.Queue(maxsize=60)

    # sel_cats = cat2imgpath.keys()
    # sel_cats = random.shuffle(sel_cats)

    for cat in cat2imgpath:
        random.shuffle(cat2imgpath[cat])
    # imgs = []
    while True:
        if not cat2imgpath:
            break
        else:
            rgb = torch.zeros(batch_size*group_size, 3, img_size, img_size)
            cls_labels = torch.zeros(batch_size, cls_size)
            mask_labels = torch.zeros(batch_size*group_size, img_size, img_size)

            sel_cats = random.sample(cat2imgpath.keys(), min(len(cat2imgpath.keys()), batch_size))
            # print("sel order: ", sel_cats, len(sel_cats))
            # for s in sel_cats:
                # print("hi")
                # print(s," ", len(cat2imgpath[s]))
            img_n = 0
            group_n = 0
            imgs = []
            for cat in sel_cats:
                # imgs.append([])
                for i in range(group_size):
                    # print("curr status: ", cat2imgpath.keys())
                    img_path = cat2imgpath[cat][0][0]

                    img = Image.open(img_path)
                    if img.mode == 'RGB':
                        img = img_transform(img)
                    else:
                        img = img_transform_gray(img)

                    mask_path = cat2imgpath[cat][0][1]
                    mask = Image.open(mask_path)

                    mask = ImageOps.grayscale(mask)
                    # mask[mask > 0] = 255
                    mask = gt_transform(mask)
                    mask[mask > 0.5] = 1
                    mask[mask <= 0.5] = 0

                    # print(img_n)
                    # print(img_path)
                    # if img.shape[0]!=3:
                    # print(img_path, cat)
                    rgb[img_n,:,:,:] = copy.deepcopy(img)
                    mask_labels[img_n,:,:] = copy.deepcopy(mask)

                    # delete image

                    cat2imgpath[cat].remove(cat2imgpath[cat][0])

                    if len(cat2imgpath[cat]) == 0:
                        del cat2imgpath[cat]
                        break

                    img_n += 1
                    imgs.append(img_path)
                # print(cat,cat2index[cat]-1)
                cls_labels[group_n, cat2index[cat]-1] = 1
                group_n += 1

            q.put([rgb, cls_labels, mask_labels])
    return q

if __name__ == "__main__":
    q = co_skel_data_producer("./final.csv")
    print(q.get())
    print("yay")