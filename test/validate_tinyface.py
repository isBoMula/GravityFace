import matplotlib.pyplot as plt
from prettytable import PrettyTable
from eval_summary import write_eval_section
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import sys, os
from PIL import Image

# 引入 PyTorch 原生的数据加载和预处理模块
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import tinyface_helper

sys.path.insert(0, os.path.dirname(os.getcwd()))

# 替换为你提供的 IJB 脚本里的模型导入方式
from backbones import get_model


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def l2_norm(input, axis=1):
    """l2 normalize"""
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output, norm


def fuse_features_with_norm(stacked_embeddings, stacked_norms, fusion_method='norm_weighted_avg'):
    assert stacked_embeddings.ndim == 3  # (n_features_to_fuse, batch_size, channel)
    if stacked_norms is not None:
        assert stacked_norms.ndim == 3  # (n_features_to_fuse, batch_size, 1)
    else:
        assert fusion_method not in ['norm_weighted_avg', 'pre_norm_vector_add']

    if fusion_method == 'norm_weighted_avg':
        weights = stacked_norms / stacked_norms.sum(dim=0, keepdim=True)
        fused = (stacked_embeddings * weights).sum(dim=0)
        fused, _ = l2_norm(fused, axis=1)
        fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'pre_norm_vector_add':
        pre_norm_embeddings = stacked_embeddings * stacked_norms
        fused = pre_norm_embeddings.sum(dim=0)
        fused, fused_norm = l2_norm(fused, axis=1)
    elif fusion_method == 'average':
        fused = stacked_embeddings.sum(dim=0)
        fused, _ = l2_norm(fused, axis=1)
        if stacked_norms is None:
            fused_norm = torch.ones((len(fused), 1))
        else:
            fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'concat':
        fused = torch.cat([stacked_embeddings[0], stacked_embeddings[1]], dim=-1)
        if stacked_norms is None:
            fused_norm = torch.ones((len(fused), 1))
        else:
            fused_norm = stacked_norms.mean(dim=0)
    elif fusion_method == 'faceness_score':
        raise ValueError('not implemented yet.')
    else:
        raise ValueError('not a correct fusion method', fusion_method)

    return fused, fused_norm


# ----------------- 新增：原生的 Dataset 实现 -----------------
class TinyFaceDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        # 定义标准的 InsightFace/AdaFace 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),  # 确保尺寸为 112x112
            transforms.ToTensor(),  # 转换为 Tensor，范围 [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # 使用 PIL 读取为 RGB 格式
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            img_tensor = torch.zeros((3, 112, 112))  # 异常时的兜底处理

        return img_tensor, idx


# -------------------------------------------------------------


def infer(model, dataloader, use_flip_test, fusion_method):
    model.eval()
    features = []
    norms = []
    with torch.no_grad():
        for images, idx in tqdm(dataloader):
            images = images.cuda()
            feature = model(images)

            if isinstance(feature, tuple):
                feature, norm = feature
            else:
                # 【修改点 1】：计算 unnormalized feature 的 norm，并将 feature 归一化
                feature, norm = l2_norm(feature, axis=1)

            if use_flip_test:
                fliped_images = torch.flip(images, dims=[3])
                flipped_feature = model(fliped_images)

                if isinstance(flipped_feature, tuple):
                    flipped_feature, flipped_norm = flipped_feature
                else:
                    # 【修改点 2】：同样处理翻转后的特征
                    flipped_feature, flipped_norm = l2_norm(flipped_feature, axis=1)

                stacked_embeddings = torch.stack([feature, flipped_feature], dim=0)
                if norm is not None:
                    stacked_norms = torch.stack([norm, flipped_norm], dim=0)
                else:
                    stacked_norms = None

                fused_feature, fused_norm = fuse_features_with_norm(stacked_embeddings, stacked_norms,
                                                                    fusion_method=fusion_method)
                features.append(fused_feature.cpu().numpy())
                norms.append(fused_norm.cpu().numpy() if fused_norm is not None else None)
            else:
                features.append(feature.cpu().numpy())
                norms.append(norm.cpu().numpy() if norm is not None else None)

    features = np.concatenate(features, axis=0)
    if norms[0] is not None:
        norms = np.concatenate(norms, axis=0)
    else:
        norms = None
    return features, norms


# ----------------- 模型加载逻辑 -----------------
def load_pretrained_model(model_path, network, embedding_size):
    print(f"Loading model from: {model_path}")
    weight = torch.load(model_path, weights_only=True)

    # 参照 IJB 脚本，调用 get_model 构建网络，送入 cuda()
    resnet = get_model(network, dropout=0, fp16=False, num_features=embedding_size).cuda()

    # [兼容性处理] 防止你的 model.pt 存的带外层包裹，按需解包
    if 'state_dict' in weight:
        weight = weight['state_dict']
    if any(key.startswith('model.') for key in weight.keys()):
        weight = {key[6:]: val for key, val in weight.items() if key.startswith('model.')}
    if any(key.startswith('module.') for key in weight.keys()):
        weight = {key[7:]: val for key, val in weight.items() if key.startswith('module.')}

    resnet.load_state_dict(weight)

    # 使用 DataParallel 封装
    model = torch.nn.DataParallel(resnet)
    model.eval()
    return model


# -------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tinyface')
    parser.add_argument('--data_root', default=r"E:\Dataset\TinyFace")
    parser.add_argument('--batch_size', default=512, type=int)

    # 修改：接收主控脚本传来的参数
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--network', type=str, default='r50', help='Network architecture')
    # 👇 加上这一行 👇
    parser.add_argument('--embedding_size', type=int, default=512, help='Feature embedding size')


    parser.add_argument('--use_flip_test', type=str2bool, default='True')
    parser.add_argument('--fusion_method', type=str, default='pre_norm_vector_add',
                        choices=('average', 'norm_weighted_avg', 'pre_norm_vector_add', 'concat', 'faceness_score'))
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    model = load_pretrained_model(args.model_path, args.network, args.embedding_size)
    tinyface_test = tinyface_helper.TinyFaceTest(tinyface_root=args.data_root,
                                                 alignment_dir_name='aligned_pad_0.1_pad_high')

    # 修改：结果统一保存至 result_dir
    save_path = args.result_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    img_paths = tinyface_test.image_paths
    dataset = TinyFaceDataset(img_paths)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    features, norms = infer(model, dataloader, use_flip_test=args.use_flip_test, fusion_method=args.fusion_method)
    results = tinyface_test.test_identification(features, ranks=[1, 5, 20])

    # 1. 保存 CSV
    pd.DataFrame({'rank': [1, 5, 20], 'values': results}).to_csv(os.path.join(save_path, 'tinyface_result.csv'))

    # 2. 绘制并保存图片快照 (类似 IJBB)
    table = PrettyTable(['Rank', 'Identification Rate (%)'])
    ranks = [1, 5, 20]
    for r, val in zip(ranks, results):
        table.add_row([f"Rank {r}", f"{val * 100:.2f}"])

    print(table)

    fig = plt.figure(figsize=(6, 3), dpi=200)
    ax = fig.add_subplot(111)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # 渲染表格为图片
    plt.text(0.05, 0.95, table.get_string(), fontsize=12, family='monospace', verticalalignment='top')
    output_file = os.path.join(save_path, "tinyface_result_table.png")
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"TinyFace 结果快照已保存至: {output_file}")
    eval_md_file = write_eval_section(save_path, "TinyFace", "TinyFace", table.get_string())
    print(f"Evaluation summary saved to: {eval_md_file}")
