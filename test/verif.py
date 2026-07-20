import os.path

import bcolz
import numpy as np
import torch
import torchvision.transforms as transforms

from backbones import get_model

import argparse
from eval_summary import write_eval_section


def Calculate_features(batch_imgs):
    batch_features = model(batch_imgs.to(DEVICE)).cpu()
    return batch_features


def Calculate_Accuracy(features, isSame):
    m = features.shape[0] // 2
    features1 = torch.tensor(features[0::2]).to(DEVICE)
    features2 = torch.tensor(features[1::2]).to(DEVICE)
    norm1 = torch.norm(features1, dim=1)
    norm2 = torch.norm(features2, dim=1)
    # norm1_mean = norm1.mean()
    # norm2_mean = norm2.mean()
    cosine = torch.sum(features1 * features2, dim=1) / (norm1 * norm2)
    # figure = torch.sum(features1 * features2, dim=1)
    # sigmoid_norm1 = 1 / (1 + torch.exp(-(norm1-norm1_mean)))
    # sigmoid_norm2 = 1 / (1 + torch.exp(-(norm2 - norm2_mean)))
    # figure = figure * sigmoid_norm1 * sigmoid_norm2

    # l2_distance = torch.sqrt(torch.sum((features1 - features2) ** 2, dim=1))
    # features1 = features1/norm1.unsqueeze(1)
    # features2 = features2 / norm2.unsqueeze(1)
    # figure = torch.sqrt(torch.sum((features1 - features2) ** 2, dim=1))

    # x_index = torch.where(figure > 0.5)[0]
    # print(x_index.shape)
    # print(figure[x_index])
    # figure[x_index] = figure[x_index] * norm1[x_index] * norm2[x_index]
    # print(figure[x_index])

    best_acc = 0
    best_th = 1
    labels = torch.tensor(isSame).to(DEVICE)
    for i in range(m):
        th = cosine[i]
        acc = ((cosine > th) == labels).int().sum() / m
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return best_th, best_acc


def Evaluate(testset, result_dir):
    idx = 0
    carray = bcolz.carray(rootdir=os.path.join(DATA_ROOT, testset), mode='r')
    isSame = np.load(DATA_ROOT + f'\\{testset}_list.npy')
    features = np.zeros([len(carray), EMBEDDING_SIZE])
    with torch.no_grad():
        while idx + BATCH_SIZE <= len(carray):
            batch = torch.tensor(carray[idx:idx + BATCH_SIZE][:, [2, 1, 0], :, :])
            features[idx:idx + BATCH_SIZE] = Calculate_features(batch)

            idx += BATCH_SIZE
        if idx < len(carray):
            batch = torch.tensor(carray[idx:][:, [2, 1, 0], :, :])
            features[idx:idx + BATCH_SIZE] = Calculate_features(batch)
    th, acc = Calculate_Accuracy(features, isSame)
    print(f"{testset}: acc = {acc:.4f}, the = {th:.4f}")
    result_line = f"{testset}: acc = {acc:.4f} th = {th:.4f}"

    file_path = os.path.join(result_dir, TXT_NAME + ".txt")
    # 使用 "a" 模式追加写入，并加上换行符 \n
    with open(file_path, "a") as f:
        f.write(result_line + "\n")
    f.close()
    return result_line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--network', type=str, default='r50')
    args = parser.parse_args()

    DATA_ROOT = r"E:\Dataset\evo"
    DEVICE = 'cuda'
    INPUT_SIZE = [112, 112]
    EMBEDDING_SIZE = 512
    BATCH_SIZE = 512
    TXT_NAME = "Accuracy"

    # 动态加载模型
    weight = torch.load(args.model_path, weights_only=True)
    resnet = get_model(args.network, dropout=0, fp16=False, num_features=512).cuda()
    resnet.load_state_dict(weight)
    model = torch.nn.DataParallel(resnet)
    model.eval()

    # 动态将 evaluate 的输出结果保存至对应的结果文件夹
    # 注意：需要同步修改你的 Evaluate 函数中的文件写入路径为：
    # f = open(os.path.join(args.result_dir, TXT_NAME + ".txt"), "a")

    verification_results = [
        Evaluate('lfw', args.result_dir),
        Evaluate('calfw', args.result_dir),
        Evaluate('cplfw', args.result_dir),
        Evaluate('agedb_30', args.result_dir),
        Evaluate('cfp_fp', args.result_dir),
        Evaluate('cfp_ff', args.result_dir),
        Evaluate('vgg2_fp', args.result_dir),
    ]
    eval_md_file = write_eval_section(args.result_dir, "Verification", "Verification", "\n".join(verification_results))
    print(f"Evaluation summary saved to: {eval_md_file}")

