# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
可视化特征提取结果
"""

import argparse
import logging
import sys

import numpy as np
import torch
import tqdm
from torch.backends import cudnn

sys.path.append('.')

import torch.nn.functional as F
from fastreid.evaluation.rank import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer

# import some modules added in project
# for example, add partial reid like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")

logger = logging.getLogger('fastreid.visualize_result')


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
        '--parallel',  # 是否并行
        action='store_true',
        help='if use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--dataset-name",  # 数据集名字
        help="a test dataset name for visualizing ranking list."
    )
    parser.add_argument(
        "--output",  # 输出结果路径
        default="./vis_rank_list",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--vis-label",  # 输出结果是否查看
        action='store_true',
        help="if visualize label of query instance"
    )
    parser.add_argument(
        "--num-vis",  # 挑选多少张图像用于结果展示
        default=1000,
        help="number of query images to be visualized",
    )
    parser.add_argument(
        "--rank-sort",  # 结果展示是相似度排序方式，默认从小到大排序
        default="ascending",
        help="rank order of visualization images by AP metric",
    )
    parser.add_argument(
        "--label-sort",  # label结果展示是相似度排序方式，默认从小到大排序
        default="ascending",
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--max-rank",  # 显示topk的结果，默认显示前10个结果
        default=5,
        help="maximum number of rank list to be visualized",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    # 调试使用，使用的时候删除下面代码
    # ---
    args.config_file = "./configs/Market1501/bagtricks_R50.yml"  # config路径
    args.dataset_name = 'Market1501'  # 数据集名字
    args.vis_label = False  # 是否显示正确label结果
    args.rank_sort = 'descending'  # 从大到小显示关联结果
    args.label_sort = 'descending'  # 从大到小显示关联结果
    # ---

    cfg = setup_cfg(args)
    # 可以直接在代码中设置cfg中设置模型路径
    # cfg["MODEL"]["WEIGHTS"] = './configs/Market1501/bagtricks_R50.yml'
    test_loader, num_query = build_reid_test_loader(cfg, dataset_name=args.dataset_name)  # 创建测试数据集
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)  # 加载特征提取器，也就是加载模型

    logger.info("Start extracting image features")
    feats = []  # 图像特征，用于保存每个行人的图像特征
    pids = []  # 行人id，用于保存每个行人的id
    camids = []  # 拍摄的摄像头，行人出现的摄像头id
    # 逐张保存读入行人图像，并保存相关信息
    for (feat, pid, camid) in tqdm.tqdm(demo.run_on_loader(test_loader), total=len(test_loader)):
        feats.append(feat)
        pids.extend(pid)
        camids.extend(camid)

    feats = torch.cat(feats, dim=0)  # 将feats转换为tensor的二维向量，向量维度为[图像数，特征维度]
    # 这里把query和gallery数据放在一起了，需要切分query和gallery的数据
    q_feat = feats[:num_query]
    g_feat = feats[num_query:]
    q_pids = np.asarray(pids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])

    # compute cosine distance 计算余弦距离
    q_feat = F.normalize(q_feat, p=2, dim=1)
    g_feat = F.normalize(g_feat, p=2, dim=1)
    distmat = 1 - torch.mm(q_feat, g_feat.t())  # 这里distmat表示两张图像的距离，越小越接近
    distmat = distmat.numpy()

    # 计算各种评价指标 cmc[0]就是top1精度，应该是93%左右，这里精度会有波动
    logger.info("Computing APs for all query images ...")
    cmc, all_ap, all_inp = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("Finish computing APs for all query images!")

    visualizer = Visualizer(test_loader.dataset)  # 创建Visualizer类
    visualizer.get_model_output(all_ap, distmat, q_pids, g_pids, q_camids, g_camids)  # 保存结果

    logger.info("Start saving ROC curve ...")  # 保存ROC曲线
    fpr, tpr, pos, neg = visualizer.vis_roc_curve(args.output)
    visualizer.save_roc_info(args.output, fpr, tpr, pos, neg)
    logger.info("Finish saving ROC curve!")

    logger.info("Saving rank list result ...")  # 保存部分查询图像的关联结果，按照顺序排列
    query_indices = visualizer.vis_rank_list(args.output, args.vis_label, args.num_vis,
                                             args.rank_sort, args.label_sort, args.max_rank)
    logger.info("Finish saving rank list results!")
