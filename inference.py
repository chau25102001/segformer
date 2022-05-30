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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--pretrain-path", type=str, default="runs/exp/best_model.pth")
    parser.add_argument("--test-path", type=str, default="/home/s/gianglt/mdeq/TestDataset/CVC-300,"
                                                         "/home/s/gianglt/mdeq/TestDataset/CVC-ClinicDB,"
                                                         "/home/s/gianglt/mdeq/TestDataset/CVC-ColonDB,"
                                                         "/home/s/gianglt/mdeq/TestDataset/ETIS-LaribPolypDB,"
                                                         "/home/s/gianglt/mdeq/TestDataset/Kvasir")
    return parser.parse_args()


def main():
    config = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    test_paths = config.test_path.split(',')
    results = {}
    for path in test_paths:
        test_set = PolypDataset(
            root_path=path,
            img_subpath="images",
            label_subpath="masks",
            img_size=config.img_size,
            use_aug=False,
            use_cutmix=0.0
        )
        test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=1)
        model = SegFormer(
            f"MiT-B{config.model_config}",
            config.number_classes,
        )
        model.load_state_dict(torch.load(config.pretrain_path, map_location=device)["model_state_dict"])
        model.to(device)
        model.eval()
        criterion = GiangPolyCriterion()
        test_bce, test_tve, test_iou, dice_iou = Trainer.evaluation(model, test_loader, device, criterion)
        results[path.split("/")[-1]] = {
            "test_bce": test_bce,
            "test_tve": test_tve,
            "test_iou": test_iou,
            "dice_iou": dice_iou
        }

    for path, result in results.items():
        print(f"{path} :-> IoU: {result['test_iou']}, Dice: {result['dice_iou']}")


if __name__ == '__main__':
    main()
