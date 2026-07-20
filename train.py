import os
import shutil
# from datetime import datetime
import random
import numpy as np
import argparse

import torch
from backbones import get_model
from dataset import my_ImageFolder
from losses import CombinedMarginLoss, ArcFace, CosFace, NaiveFace, PFace,SFace, AdaFace, CurricularFace
# from lr_scheduler import PolynomialLRWarmup
from lr_scheduler import MHLR
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, CosineAnnealingLR
from torchvision import transforms
from torchvision.datasets import ImageFolder
from partial_fc_v2 import DualViewGravitationalLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from logger import logger
import importlib

def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True



def main(config_file):
    # get config
    config = importlib.import_module("configs."+config_file)
    cfg = config.cfg()
    
    device = torch.device(cfg.device)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    os.makedirs(cfg.output, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
    log = logger(cfg=cfg, start_step = 0, writer=summary_writer)
    
    # Image Folder
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
    train_set = my_ImageFolder(cfg.rec, transform)
    train_loader = DataLoader(dataset=train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size)
    backbone.train().to(device)

    # margin_loss = ArcFace()

    # margin_loss = CurricularFace(512, 93431)
    CE_loss = DualViewGravitationalLoss(cfg.embedding_size, cfg.num_classes)

    CE_loss.train().to(device)

    opt = torch.optim.SGD(params=[{"params": backbone.parameters()}, {"params": CE_loss.parameters()}], lr=cfg.lr,
                          momentum=0.9, weight_decay=cfg.weight_decay)
    # opt = torch.optim.Adam(params=[{"params": backbone.parameters()}, {"params": CE_loss.parameters()}], lr=cfg.lr, weight_decay=cfg.weight_decay)
    # opt = torch.optim.AdamW(params=[{"params": backbone.parameters()}, {"params": CE_loss.parameters()}], lr=cfg.lr, weight_decay=cfg.weight_decay)

    lr_scheduler = CosineAnnealingLR(opt, T_max=cfg.total_step)


    global_step = 0
    try:
        # PyTorch 2.0+
        from torch.amp import GradScaler
        amp = GradScaler("cuda", growth_interval=100)
    except ImportError:
        # PyTorch 1.6 - 1.13
        from torch.cuda.amp import GradScaler
        amp = GradScaler(growth_interval=100)
    for epoch in range(0, cfg.num_epoch):
        for _, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img.to(device))

            loss: torch.Tensor = CE_loss(local_embeddings, local_labels.to(device))

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                log(global_step, loss.item(), epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.state_dict(),
                "state_dict_softmax_fc": CE_loss.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{epoch}.pt"))


    path_module = os.path.join(cfg.output, "model.pt")
    torch.save(backbone.state_dict(), path_module)
    # 保存FC层参数
    path_module = os.path.join(cfg.output, "fc.pt")
    torch.save(CE_loss.state_dict(), path_module)
    log.loss2csv(cfg.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get configurations')
    # parser.add_argument('--config', default="mnist", help='the name of config file')
    parser.add_argument('--config', default="ms1mv3", help='the name of config file')
    args = parser.parse_args()
    main(args.config)
    # 训练完成后进行准确率测试
    # import eval_ijbb
    # import eval_ijbc
