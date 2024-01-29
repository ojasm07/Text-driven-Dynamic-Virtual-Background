import torch
import argparse
from data import MP3Ddataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from main import PanoGenerator
from params import params


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

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.set_float32_matmul_precision('medium')

    params_ = params()
    params_.max_epochs = args.max_epochs
    params_.batch_size = args.batch_size

    if params_.dataset_name == 'mp3d':
        train_dataset = MP3Ddataset(mode='train')
        val_dataset = MP3Ddataset(mode='val')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params_.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)

    if params_.model_type == 'pano_generation':
        model = PanoGenerator()

    if args.ckpt_path is not None:
        model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu')[
            'state_dict'], strict=False)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="train_loss",
                                          mode="min", save_last=1,
                                          filename='epoch={epoch}-loss={train_loss:.4f}')

    logger = TensorBoardLogger(
        save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        logger=logger)

    trainer.fit(model, train_loader, val_loader)
