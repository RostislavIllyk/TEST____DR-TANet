from TANet import TANet
import os

import cv2
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset
from os.path import join as pjoin, splitext as spt


class DataLoader(Dataset):

    def __init__(self, root):
        super(DataLoader, self).__init__()

        self.img_t0_root = root + '/t0'
        self.img_t1_root = root + '/t1'

        self.filename = list(spt(f)[0] for f in os.listdir(self.img_t0_root) if check_validness(f))
        self.filename.sort()

    def __getitem__(self, index):

        fn = self.filename[index]
        fn_t0 = pjoin(self.img_t0_root, fn + '.jpg')
        fn_t1 = pjoin(self.img_t1_root, fn + '.jpg')

        if os.path.isfile(fn_t0) is False:
            print('Error: File Not Found: ' + fn_t0)
            exit(-1)
        if os.path.isfile(fn_t1) is False:
            print('Error: File Not Found: ' + fn_t1)
            exit(-1)

        img_t0 = cv2.imread(fn_t0, 1)
        img_t1 = cv2.imread(fn_t1, 1)

        w, h, c = img_t0.shape
        w_r = int(256 * max(w / 256, 1))
        h_r = int(256 * max(h / 256, 1))

        # resize images so that min(w, h) == 256
        img_t0_r = cv2.resize(img_t0, (h_r, w_r))
        img_t1_r = cv2.resize(img_t1, (h_r, w_r))

        img_t0_r = np.asarray(img_t0_r).astype('f').transpose(2, 0, 1) / 128.0 - 1.0
        img_t1_r = np.asarray(img_t1_r).astype('f').transpose(2, 0, 1) / 128.0 - 1.0

        return img_t0_r, img_t1_r, w, h, w_r, h_r

    def __len__(self):
        return len(self.filename)


def check_validness(f):
    return any([i in spt(f)[1] for i in ['jpg', 'png']])


class Prediction:

    def __init__(self, arguments, model_path):
        self.args = None
        self.set = None
        self.t0 = None
        self.t1 = None
        self.w_ori = None
        self.h_ori = None
        self.w_r = None
        self.h_r = None
        self.fn_img = None
        self.fn_model = None
        self.dir_img = None
        self.model = None
        self.resultdir = None

        self.args = arguments
        self.fn_model = model_path

    def predict(self):

        input = torch.from_numpy(np.concatenate((self.t0, self.t1), axis=0)).contiguous()
        input = input.view(1, -1, self.w_r, self.h_r)
        input = input.cuda()
        output = self.model(input)

        output = output[0].cpu().data

        mask_pred = np.where(F.softmax(output[0:2, :, :], dim=0)[0] > 0.5, 255, 0)

        if self.args.store_imgs:
            self.store_imgs_and_cal_matrics(mask_pred)
        else:
            pass
        return

    def store_imgs_and_cal_matrics(self, mask_pred):

        w, h = self.w_r, self.h_r

        img_save = cv2.cvtColor(mask_pred.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        if w != self.w_ori or h != self.h_ori:
            img_save = cv2.resize(img_save, (self.h_ori, self.w_ori))

        fn_save = self.fn_img
        if not os.path.exists(self.dir_img):
            os.makedirs(self.dir_img)

        print('Writing' + fn_save + '......')

        cv2.imwrite(fn_save, img_save)

        return

    def init(self):

        if self.args.drtam:
            print('Dynamic Receptive Temporal Attention Network (DR-TANet)')
            model_name = 'DR-TANet'
        else:
            print('Temporal Attention Network (TANet)')
            model_name = 'TANet_k={0}'.format(self.args.local_kernel_size)

        model_name += ('_' + self.args.encoder_arch)

        print('Encoder:' + self.args.encoder_arch)

        if self.args.refinement:
            print('Adding refinement...')
            model_name += '_ref'

        self.resultdir = pjoin(self.args.resultdir, model_name, self.args.dataset)
        if not os.path.exists(self.resultdir):
            os.makedirs(self.resultdir)
        self.dir_img = self.resultdir + '/imgs'

    def run(self):

        if os.path.isfile(self.fn_model) is False:
            print("Error: Cannot read file ... " + self.fn_model)
            exit(-1)
        else:
            print("Reading model ... " + self.fn_model)

        self.model = TANet(self.args.encoder_arch, self.args.local_kernel_size, self.args.attn_stride,
                           self.args.attn_padding, self.args.attn_groups, self.args.drtam, self.args.refinement)

        if self.args.multi_gpu:
            self.model = nn.DataParallel(self.model)

        self.model.load_state_dict(torch.load(self.fn_model))
        self.model = self.model.cuda()

        test_loader = DataLoader(self.args.datadir)

        img_cnt = len(test_loader)
        for idx in range(0, img_cnt):
            index = idx
            ds = 'result'
            self.fn_img = self.dir_img + '/{0}-{1:08d}.png'.format(ds, index)
            self.t0, self.t1, self.w_ori, self.h_ori, self.w_r, self.h_r = test_loader[idx]
            self.predict()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='STRAT PREDICTING...')
    parser.add_argument('--dataset', type=str, default='pcd')
    parser.add_argument('--datadir', default='./files_to_predict')
    parser.add_argument('--resultdir', default='./predicted')
    parser.add_argument('--checkpointdir', default='./check')
    parser.add_argument('--encoder-arch', type=str,   default='resnet18')
    parser.add_argument('--local-kernel-size', type=int, default=1)
    parser.add_argument('--attn-stride', type=int, default=1)
    parser.add_argument('--attn-padding', type=int, default=0)
    parser.add_argument('--attn-groups', type=int, default=4)
    parser.add_argument('--drtam', action='store_true', default=True)
    parser.add_argument('--refinement', action='store_true', default=True)
    parser.add_argument('--store-imgs', action='store_true', default=True)
    parser.add_argument('--multi-gpu', action='store_true', help='processing with multi-gpus')

    path_to_model = './check/DR-TANet_resnet18_ref/checkpointdir/00040000.pth'
    # path_to_model = './check/DR-TANet_resnet34_ref/checkpointdir/00009000.pth'

    if parser.parse_args().dataset == 'pcd':
        predict = Prediction(parser.parse_args(), path_to_model)
        predict.init()
        predict.run()

    else:
        print('Error: Cannot identify the dataset...(dataset: pcd or vl_cmu_cd)')
        exit(-1)
