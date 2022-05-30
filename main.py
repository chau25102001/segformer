import os
import torch
import argparse
from utils.generals import intersect_dicts
from losses.criterion import GiangPolyCriterion
from trainer.trainer import Trainer
from dataset.polyp import PolypDataset
from models.segformer import SegFormer
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config', type=int, default=3)
    parser.add_argument("--number-classes", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save-path", type=str, default="runs/exp")
    parser.add_argument("--train-path", type=str, default="/home/s/gianglt/mdeq/TrainDataset/train")
    parser.add_argument("--test-path", type=str, default="/home/s/gianglt/mdeq/TestDataset/CVC-300,"
                                                         "/home/s/gianglt/mdeq/TestDataset/CVC-ClinicDB,"
                                                         "/home/s/gianglt/mdeq/TestDataset/CVC-ColonDB,"
                                                         "/home/s/gianglt/mdeq/TestDataset/ETIS-LaribPolypDB,"
                                                         "/home/s/gianglt/mdeq/TestDataset/Kvasir")
    parser.add_argument("--max-norm", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--use-tensorboard", action="store_true")
    return parser.parse_args()


def main():
    config = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train_set = PolypDataset(
        root_path=config.train_path,
        img_subpath="image",
        label_subpath="mask",
        img_size=config.img_size,
        use_aug=True,
        use_cutmix=0.5
    )
    test_set = PolypDataset(
        root_path=config.test_path.split(","),
        img_subpath="images",
        label_subpath="masks",
        img_size=config.img_size,
        use_aug=False,  
        use_cutmix=0.0
    )
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=8)
    model = SegFormer(
        f"MiT-B{config.model_config}",
        config.number_classes,
    )
    ckpt = torch.load("runs/exp/last_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    # ckpt = intersect_dicts(torch.load("mit_b3.pth", map_location=device), model.backbone.state_dict())
    # print(len(ckpt.keys()))
    # model.backbone.load_state_dict(ckpt, strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=0.0001,
        total_iters=3 * config.epochs * len(train_set) // config.batch_size
    )
    criterion = GiangPolyCriterion()
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        criterion=criterion,
        use_tensorboard=config.use_tensorboard,
        epochs=config.epochs,
        device=device,
        save_dir=config.save_path,
        exist_ok=config.exist_ok,
        max_norm=config.max_norm
    )
    trainer.train()


if __name__ == '__main__':
    main()
