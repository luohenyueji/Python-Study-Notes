# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
提取图像的特征，并保存
"""

import argparse
import glob
import os
import sys

import torch.nn.functional as F
import cv2
import numpy as np
import tqdm
from torch.backends import cudnn

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager

from predictor import FeatureExtractionDemo

# import some modules added in project like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")


# 读取配置文件
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",  # config路径，通常包含模型配置文件
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--parallel",  # 是否并行
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input",  # 输入图像路径
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",  # 输出结果路径
        default='demo_output',
        help='path to save features'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)  # 特征归一化
    features = features.cpu().data.numpy()
    return features


if __name__ == '__main__':
    args = get_parser().parse_args()  # 解析输入参数
    # 调试使用，使用的时候删除下面代码
    # ---
    args.config_file = "./configs/Market1501/bagtricks_R50.yml"  # config路径
    args.input = "./datasets/Market-1501-v15.09.15/query/*.jpg"  # 图像路径
    # ---

    cfg = setup_cfg(args)  # 读取cfg文件
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)  # 加载特征提取器，也就是加载模型

    PathManager.mkdirs(args.output)  # 创建输出路径
    if args.input:
        if PathManager.isdir(args.input[0]):  # 判断输入的是否为路径
            # args.input = glob.glob(os.path.expanduser(args.input[0])) # 原来的代码有问题
            args.input = glob.glob(os.path.expanduser(args.input))  # 获取输入路径下所有的文件路径
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input):  # 逐张处理
            img = cv2.imread(path)
            feat = demo.run_on_image(img)  # 提取图像特征
            feat = postprocess(feat)  # 后处理主要是特征归一化
            np.save(os.path.join(args.output, os.path.basename(path).split('.')[0] + '.npy'), feat)  # 保存图像对应的特征，以便下次使用
