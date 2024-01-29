import torch
import argparse
from data import MP3Ddataset
import pytorch_lightning as pl
from main import PanoGenerator
from pytorch_lightning.loggers import TensorBoardLogger
from params import params
from weights.download_wts import download_wts
import os


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=0)
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')
    parser.add_argument(
        '--mode', type=str, default='val',
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')
    parser.add_argument(
        '--eval_on_train', action='store_true')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.set_float32_matmul_precision('medium')
    params_ = params()
    params_.max_epochs = args.max_epochs

    image_root_dir = "training/mp3d_skybox"

    mode = 'train' if args.eval_on_train else 'val'

    if params_.dataset_name == 'mp3d':
        dataset = MP3Ddataset(mode=mode)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)

    if params_.model_type == 'pano_generation':
        model = PanoGenerator()

    if args.ckpt_path is not None:
        model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu')[
            'state_dict'], strict=True)
    else:
        if params_.model_ckpt_path and not os.path.exists(params_.model_ckpt_path):
            download_wts()

        model.load_state_dict(torch.load(params_.model_ckpt_path, map_location='cpu')[
            'state_dict'], strict=True)

    logger = TensorBoardLogger(
        save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger)

    if args.mode == 'test':
        trainer.test(model, data_loader)
    else:
        trainer.validate(model, data_loader)
