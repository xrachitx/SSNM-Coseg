from loss import Loss
from torch.optim import Adam
from tools import custom_print
import datetime
import torch
from val import validation

def train(net, device, q, log_txt_file, val_datapath, models_train_best, models_train_last, lr=1e-4, lr_de_epoch=25000,
          epochs=100000, log_interval=100, val_interval=1000):

    optimizer = Adam(net.parameters(), lr, weight_decay=1e-6)
    loss = Loss().to(device)
    best_p, best_j = 0, 1
    ave_loss, ave_m_loss, ave_c_loss, ave_s_loss = 0, 0, 0, 0
    print("Starting training")
    for epoch in range(1, epochs+1):
        img, cls_gt, mask_gt = q.get()
        net.zero_grad()
        img, cls_gt, mask_gt = img.to(device), cls_gt.to(device), mask_gt.to(device)
        pred_cls, pred_mask = net(img)
        all_loss, m_loss, c_loss, s_loss = loss(pred_mask, mask_gt, pred_cls, cls_gt)
        all_loss.backward()
        epoch_loss = all_loss.item()
        m_l = m_loss.item()
        c_l = c_loss.item()
        s_l = s_loss.item()
        ave_loss += epoch_loss
        ave_m_loss += m_l
        ave_c_loss += c_l
        ave_s_loss += s_l
        optimizer.step()
        if epoch % log_interval == 0:
            ave_loss = ave_loss / log_interval
            ave_m_loss = ave_m_loss / log_interval
            ave_c_loss = ave_c_loss / log_interval
            ave_s_loss = ave_s_loss / log_interval
            custom_print(datetime.datetime.now().strftime('%F %T') +
                         ' lr: %e, epoch: [%d/%d], all_loss: [%.4f], m_loss: [%.4f], c_loss: [%.4f], s_loss: [%.4f]' %
                         (lr, epoch, epochs, ave_loss, ave_m_loss, ave_c_loss, ave_s_loss), log_txt_file, 'a+')
            ave_loss, ave_m_loss, ave_c_loss, ave_s_loss = 0, 0, 0, 0

        if epoch % val_interval == 0:
            net.eval()
            with torch.no_grad():
                custom_print(datetime.datetime.now().strftime('%F %T') +
                             ' now is evaluating the coseg dataset', log_txt_file, 'a+')
                ave_p, ave_j = validation(net, val_datapath, device, group_size=5, img_size=224, img_dir_name='image', gt_dir_name='groundtruth',
                                          img_ext=['.jpg', '.jpg', '.jpg', '.jpg'], gt_ext=['.png', '.bmp', '.jpg', '.png'])
                if ave_p[3] > best_p:
                    # follow yourself save condition
                    best_p = ave_p[3]
                    best_j = ave_j[0]
                    torch.save(net.state_dict(), models_train_best)
                torch.save(net.state_dict(), models_train_last)
                custom_print('-' * 100, log_txt_file, 'a+')
                custom_print(datetime.datetime.now().strftime('%F %T') + ' iCoseg8  p: [%.4f], j: [%.4f]' %
                             (ave_p[0], ave_j[0]), log_txt_file, 'a+')
                custom_print(datetime.datetime.now().strftime('%F %T') + ' MSRC7    p: [%.4f], j: [%.4f]' %
                             (ave_p[1], ave_j[1]), log_txt_file, 'a+')
                custom_print(datetime.datetime.now().strftime('%F %T') + ' Int_300  p: [%.4f], j: [%.4f]' %
                             (ave_p[2], ave_j[2]), log_txt_file, 'a+')
                custom_print(datetime.datetime.now().strftime('%F %T') + ' PAS_VOC  p: [%.4f], j: [%.4f]' %
                             (ave_p[3], ave_j[3]), log_txt_file, 'a+')
                custom_print('-' * 100, log_txt_file, 'a+')
            net.train()

        if epoch % lr_de_epoch == 0:
            optimizer = Adam(net.parameters(), lr/2, weight_decay=1e-6)
            lr = lr / 2
        # del img, cls_gt, mask_gt,all_loss, m_loss, c_loss, s_loss,pred_cls, pred_mask 
    print("Training end")