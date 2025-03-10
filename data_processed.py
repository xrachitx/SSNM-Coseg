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

def co_skel_data_producer(csv_file,batch_size=10, group_size=5, img_size=224,gt=0, max_images=50, cls_size=13):
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
        if len(cat2imgpath[row[-1]]) <= max_images:
            if gt == 0:
                cat2imgpath[row[-1]].append([row[0], row[3]])
            elif gt == 1:#Complete Seg Mask
                cat2imgpath[row[-1]].append([row[0], row[2]])
            else:#Actual Seg Masks
                cat2imgpath[row[-1]].append([row[0], row[1]])
        else:
            pass
    
    rgb = torch.zeros(batch_size*group_size, 3, img_size, img_size)
    cls_labels = torch.zeros(batch_size, cls_size)
    mask_labels = torch.zeros(batch_size*group_size, img_size, img_size)

    q = queue.Queue(maxsize=143)

    # sel_cats = cat2imgpath.keys()
    # sel_cats = random.shuffle(sel_cats)

    for cat in cat2imgpath:
        random.shuffle(cat2imgpath[cat])

    while True:
        if not cat2imgpath:
            break
        else:
            sel_cats = random.sample(cat2imgpath.keys(), batch_size)

            img_n = 0
            group_n = 0
            for cat in sel_cats:
                for i in range(group_size):
                    img_path = cat2imgpath[cat][0][0]

                    img = Image.open(img_path)
                    if img.mode == 'RGB':
                        img = img_transform(img)
                    else:
                        img = img_transform_gray(img)

                    mask_path = cat2imgpath[cat][0][1]
                    mask = Image.open(mask_path)

                    mask = ImageOps.grayscale(mask)
                    mask[mask > 0] = 255
                    mask = gt_transform(mask)
                    mask[mask > 0.5] = 1
                    mask[mask <= 0.5] = 0
                    rgb[img_n,:,:,:] = copy.deepcopy(img)
                    mask_labels[img_n,:,:] = copy.deepcopy(mask)

                    # delete image

                    cat2imgpath[cat].remove(cat2imgpath[cat][0])

                    if len(cat2imgpath[cat]) == 0:
                        del cat2imgpath[cat]

                    img_n += 1
                cls_labels[group_n, cat2index[cat]] = 1
                group_n += 1

            q.put([rgb, cls_labels, mask_labels])
    return q


# def train_data_producer(coco_item, datapath, npy, q, batch_size=10, group_size=5, img_size=224):
#     img_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
#                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#     gt_transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
#     img_transform_gray = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
#                                         transforms.Normalize(mean=[0.449], std=[0.226])])
#     if os.path.exists(npy):
#         list_dict = np.load(npy,allow_pickle=True).item()
#     else:
#         list_dict = filt_small_instance(coco_item, pixthreshold=4000, imgNthreshold=100)
#     catid2label={}
#     n=0
#     for catid in list_dict:
#         catid2label[catid] = n
#         n=n+1
#     while 1:
#         rgb = torch.zeros(batch_size*group_size, 3, img_size, img_size)
#         cls_labels = torch.zeros(batch_size, 78)
#         mask_labels = torch.zeros(batch_size*group_size, img_size, img_size)
#         if batch_size> len(list_dict):
#             remainN = batch_size - len(list_dict)
#             batch_catid = random.sample(list_dict.keys(), remainN) + random.sample(list_dict, len(list_dict))
#         else:
#             batch_catid = random.sample(list_dict.keys(), batch_size)
#         group_n = 0
#         img_n = 0
#         # print("batch: ",batch_catid)
#         for catid in batch_catid:
#             imgids = random.sample(list_dict[catid], group_size)
#             # print("Sel Images: ",imgids)
#             co_catids = []
#             anns = coco_item.imgToAnns[imgids[0]]
#             for ann in anns:
#                 if  (ann['iscrowd'] == 0) and (ann['area'] > 4000):
#                     co_catids.append(ann['category_id'])
#             co_catids_backup = copy.deepcopy(co_catids)
#             for imgid in imgids[1:]:
#                 img_catids = []
#                 anns = coco_item.imgToAnns[imgid]
#                 for ann in anns:
#                     if (ann['iscrowd'] == 0) and (ann['area'] > 4000):
#                         img_catids.append(ann['category_id'])
#                 for co_catid in co_catids_backup:
#                     if co_catid not in img_catids:
#                         co_catids.remove(co_catid)
#                 co_catids_backup = copy.deepcopy(co_catids)
#             for co_catid in co_catids:
#                 cls_labels[group_n, catid2label[co_catid]] = 1
#             for imgid in imgids:
#                 path = datapath + '%012d.jpg'%imgid
#                 img = Image.open(path)
#                 if img.mode == 'RGB':
#                     img = img_transform(img)
#                 else:
#                     img = img_transform_gray(img)
#                 anns = coco_item.imgToAnns[imgid]
#                 mask = None
#                 for ann in anns:
#                     if ann['category_id'] in co_catids:
#                         if mask is None:
#                             mask = coco_item.annToMask(ann)
#                         else:
#                             mask = mask + coco_item.annToMask(ann)
#                 mask[mask > 0] = 255
#                 mask = Image.fromarray(mask)
#                 mask = gt_transform(mask)
#                 mask[mask > 0.5] = 1
#                 mask[mask <= 0.5] = 0
#                 rgb[img_n,:,:,:] = copy.deepcopy(img)
#                 mask_labels[img_n,:,:] = copy.deepcopy(mask)
#                 img_n = img_n + 1
#             group_n = group_n + 1
#         idx = mask_labels[:, :, :] > 1
#         mask_labels[idx] = 1
#         # print(cls_labels)
#         # print("\n End 1")
#         # print(torch.unique(cls_labels, return_counts=True))
#         # print("\n End 2")
#         # print("rgb ",rgb.shape)
#         # print("cls_labels ",cls_labels.shape)
#         # print("mask_labels ",mask_labels.shape)
#         print("Q size: ",q.qsize())
#         q.put([rgb, cls_labels, mask_labels])


