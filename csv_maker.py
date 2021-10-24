import os
import csv

sel_cat = {  
        1:"Aeroplane",
        2:"Bear",
        3:"Bird",
        4:"Bus",
        5:"Cats",
        6:"Cow",
        7:"Cycle",
        8:"Dog",
        9:"Elephant",
       10:"Giraffe",
       11:"Horse",
       12:"Sheep",
       13:"Zebra"
    }
    
fin_arr = []
for i in sel_cat:
    partial = os.listdir(f".\\Mittal et al Dataset\\Complete Masks\\{sel_cat[i]}\\Partial\\/")
    noisy = os.listdir(f".\\Mittal et al Dataset\\Complete Masks\\{sel_cat[i]}\\Noisy\\/")
    occlusions = os.listdir(f".\\Mittal et al Dataset\\Complete Masks\\{sel_cat[i]}\\Occlusions\\/")

    all_images = partial + noisy + occlusions

    for img in all_images:
        img_path = f"../../input/dataset/Dataset/images/{sel_cat[i]}/{img}"
        seg_path = f"../../input/dataset/Dataset/seg_mask/{sel_cat[i]}/{img}"
        compl_mask_path = f"../../input/dataset/Dataset/complete_mask/{sel_cat[i]}/{img}"
        gt_path = f"../../input/dataset/sDataset/gt/{sel_cat[i]}/{img}"
        entry = [img_path, seg_path, compl_mask_path ,gt_path,sel_cat[i]]
        fin_arr.append(entry)
    

with open("final.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(fin_arr)