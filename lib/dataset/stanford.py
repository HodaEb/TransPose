import csv
import requests
import xml.etree.ElementTree as ET
import glob
import os
import cv2
from bs4 import BeautifulSoup
import copy
import logging
import random
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
from utils.transforms import get_affine_transform_hoda

logger = logging.getLogger(__name__)

# anno_list = glob.glob1('F:/Projects/working on stanford40/XMLAnnotations','*.xml')
# for item in anno_list:
#     xml_file = os.path.join('F:/Projects/working on stanford40/XMLAnnotations',item)
#     tree = ET.parse(xml_file)
#     root = tree.getroot()
#
#     for child in root:
#         xml_file.etree.ElementTree.fromstring(child)
#         print(child.tag, child.attrib)
#     with open(xml_file, 'r') as f:
#         data = f.read()
#     Bs_data = BeautifulSoup(data, "xml")
#     b_obj = Bs_data.find_all('object')
#
import scipy.io as scio
from torch.utils.data import Dataset


class stanford(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super(stanford).__init__()
        self.data_format = cfg.DATASET.DATA_FORMAT
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.transform = transform
        self.rotation = cfg.DATASET.ROT_FACTOR
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.is_train = is_train
        # self.Tensor = T.ToTensor()
        # self.Resize = T.Resize((224, 224))
        self.color_rgb = cfg.DATASET.COLOR_RGB
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.root_img = '/content/TransPose/data/stanford/JPEGImages'
        self.anno = scio.loadmat('/content/TransPose/data/stanford/MatlabAnnotations/annotation.mat')
        self.img_name = []
        self.images = []
        self.action = []
        self.annotations = []
        self.pixel_std = 200
        self.num_joints = 17
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.parent_ids = None
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)
        if is_train:
            with open(cfg.DATASET.train_split) as fp:
                contents = fp.read().split('\n')
                # contents = contents[:10]
        else:
            with open(cfg.DATASET.test_split) as fp:
                contents = fp.read().split('\n')
                # contents = contents[:10]
        with open('/content/TransPose/data/stanford/stanford40dataset', 'rb') as handle:
            b = pickle.load(handle)
        j = 0
        for i, img in enumerate(self.anno['annotation'][0]):
            # print(img[0])
            if img[0][0][0].item() in contents:
                self.img_name.append(img[0][0][0].item()) # image name
                self.images.append(os.path.join(self.root_img,self.img_name[j])) # image root
                key = self.img_name[j].split('.')[0][:-4]
                self.action.append(b[key]) # image action
                # self.action.append(self.img_name[i].split('.')[0][:-4]) # image action
                self.boxes = []
                for box in img[0][0][1]:
                    xmin, ymin, xmax, ymax = box
                    w = xmax - xmin + 1
                    h = ymax - ymin + 1
                    center, scale = self._xywh2cs(xmin, ymin, w, h)
                    # < xmax > 258 < / xmax >
                    # < xmin > 28 < / xmin >
                    # < ymax > 400 < / ymax >
                    # < ymin > 57 < / ymin >
                    self.boxes.append({'class': 'person',
                                       'center': center,
                                       'scale': scale
                                       })
                self.annotations.append({
                    'image_name' : self.img_name[j],
                    'image_root' : self.images[j],
                    'action' : self.action[j],
                    'center': self.boxes[0]['center'],
                    'scale': self.boxes[0]['scale'],
                    'image_size' : [0,0],

                })
                j += 1
    def __getitem__(self, idx):
        # img = cv2.imread(
        #     self.images[x], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        # )
        # self.annotations[x]['image_size'] = [img.shape[0],img.shape[1]]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        db_rec = copy.deepcopy(self.annotations[idx])

        image_file = db_rec['image_root']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
        action = db_rec['action']

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            # logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        # joints = db_rec['joints_3d']
        # joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        if self.is_train:
            r = self.rotation
        else:
            r = 0

        # if self.is_train:
        #     if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
        #             and np.random.rand() < self.prob_half_body):
        #         c_half_body, s_half_body = self.half_body_transform(
        #             joints, joints_vis
        #         )
        #
        #         if c_half_body is not None and s_half_body is not None:
        #             c, s = c_half_body, s_half_body
        #
        #     sf = self.scale_factor
        #     rf = self.rotation_factor
        #     s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        #     r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
        #         if random.random() <= 0.6 else 0
        #
        #     if self.flip and random.random() <= 0.5:
        #         data_numpy = data_numpy[:, ::-1, :]
        #         joints, joints_vis = fliplr_joints(
        #             joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
        #         c[0] = data_numpy.shape[1] - c[0] - 1

        # joints_heatmap = joints.copy()
        # trans = get_affine_transform(c, s, r, self.image_size)
        trans, inv_trans = get_affine_transform_hoda(c, s, r, self.image_size)
        trans_heatmap, inv_trans_heatmap = get_affine_transform_hoda(c, s, r, self.heatmap_size)

        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        '''this is for first trial'''
        # added by me
        # original_image = cv2.warpAffine(input,
        #                                 inv_trans,
        #                                 (int(s[0]*200), int(s[1]*200)),
        #                                 flags=cv2.INTER_CUBIC)
        '''the one which works'''
        # original_image = cv2.warpAffine(input,
        #                                 inv_trans,
        #                                 (int(data_numpy.shape[1]), int(data_numpy.shape[0])),
        #                                 flags=cv2.INTER_LINEAR)
        #
        # root = r'F:/Projects/Transpose/TransPose/inverse_transformed'
        # input_file = os.path.join(root,'transformed_' + db_rec['image_name'])
        # original_file = os.path.join(root,'original_' + db_rec['image_name'])
        #
        # cv2.imwrite(input_file,input)
        # cv2.imwrite(original_file,original_image)
        ##

        if self.transform:
            input = self.transform(input)
            # image_ = self.Tensor(data_numpy)
            # image_ = self.Resize(image_)

        # for i in range(self.num_joints):
        #     if joints_vis[i, 0] > 0.0:
        #         joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        #         joints_heatmap[i, 0:2] = affine_transform(joints_heatmap[i, 0:2], trans_heatmap)
        #
        # target, target_weight = self.generate_target(joints_heatmap, joints_vis)
        #
        # target = torch.from_numpy(target)
        # target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            # 'joints': joints,
            # 'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score,
            'target': action,
            'inverse_heatmap_transform': inv_trans_heatmap,
            # 'inverse_trans_input': inv_trans,
            # 'image_': image_,
            'size': data_numpy.shape

        }

        # return input, target, target_weight, meta
        return input, meta

        # return img, self.annotations[x]
    def __len__(self):
        return len(self.annotations)
    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + (w - 1) * 0.5
        center[1] = y + (h - 1) * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

# stan = Stanford40()
# img, anno = stan[500]

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

"""for showing the picture
it doesnt work for this new structure of ___getitem___"""

# x = np.array(Image.open(img), dtype=np.uint8)
# plt.imshow(x)
#
# # Create figure and axes
# fig, ax = plt.subplots(1)
#
# # Display the image
# ax.imshow(x)
# box = anno['boxes'][0]['bbox']
# # Create a Rectangle patch
# rect = patches.Rectangle((box[0],box[1]), box[2]-box[0], box[3]-box[1], linewidth=1,
#                          edgecolor='r',facecolor='none')
#
# # Add the patch to the Axes
# ax.add_patch(rect)
# plt.show()
