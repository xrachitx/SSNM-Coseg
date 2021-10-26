import os
import torch
from pycocotools import coco
import queue
import threading
from model import build_model, weights_init
from tools import custom_print
from data_processed2 import co_skel_data_producer
from train import train
import csv,random
import time
torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    # train_val_config
    # annotation_file = '../../input/coco-2017-dataset/coco2017/annotations/instances_train2017.json'
    # coco_item = coco.COCO(annotation_file=annotation_file)

    # train_datapath = '../../input/coco-2017-dataset/coco2017/train2017/'

    val_datapath = ['../../input/ssnm-val/datasets/iCoseg8',
                    '../../input/ssnm-val/datasets/MSRC7',
                    '../../input/ssnm-val/datasets/Internet_Datasets300',
                    '../../input/ssnm-val/datasets/PASCAL_VOC']

    vgg16_path = './vgg16_bn_feat.pth'
    # npy = './new_cat2imgid_dict4000.npy'

    # project config
    project_name = 'SSNM-Coseg'
    device = torch.device('cuda:0')
    img_size = 224
    lr = 1e-5
    lr_de = 20000
    epochs = 10000
    batch_size = 5
    group_size = 5
    log_interval = 100
    val_interval = 1000

    # create log dir
    log_root = './logs'
    if not os.path.exists(log_root):
        os.makedirs(log_root)

    # create log txt
    log_txt_file = os.path.join(log_root, project_name + '_log.txt')
    custom_print(project_name, log_txt_file, 'w')

    # create model save dir
    models_root = './models'
    if not os.path.exists(models_root):
        os.makedirs(models_root)

    models_train_last = os.path.join(models_root, project_name + '_last.pth')
    models_train_best = os.path.join(models_root, project_name + '_best.pth')

    net = build_model(device).to(device)
    net.train()
    net.apply(weights_init)
    net.base.load_state_dict(torch.load(vgg16_path))

    csv_file = "./final.csv"

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

    q = queue.Queue(maxsize=40)

    # continute load checkpoint
    # net.load_state_dict(torch.load('./models/SSNM-Coseg_last.pth', map_location='cuda:0'))


    p1 = threading.Thread(target=co_skel_data_producer, args=(cat2imgpath,q))
    p2 = threading.Thread(target=co_skel_data_producer, args=(cat2imgpath,q))
    p3 = threading.Thread(target=co_skel_data_producer, args=(cat2imgpath,q))
    p1.start()
    p2.start()
    p3.start()
    time.sleep(2)

    # q = co_skel_data_producer("./final.csv",batch_size,group_size)
    train(net, device, q, log_txt_file, val_datapath, models_train_best, models_train_last, lr, lr_de, epochs, log_interval, val_interval)



