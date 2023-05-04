import argparse
import glob
import numpy as np
import os
import pprint
import torch
import torchvision
import tqdm

from glob import glob
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from utils import load_txt, accuracy, create_barplot, get_fname, AverageMeter
from models.resnet import *
from dataset import CIFAR10C


CORRUPTIONS = load_txt('./src/corruptions.txt')
MEAN = [0.49139968, 0.48215841, 0.44653091]
STD  = [0.24703223, 0.24348513, 0.26158784]


def main(opt, weight_path :str):

    device = torch.device(opt.gpu_id)

    # model
    x = torch.rand((1, 3, 32, 32)).to(device)
    model = ResNet50(c=0, num_classes=10, norm_layer=opt.norm_layer, device=device, mod=opt.mod, fc_sn=False).to(device)
    with torch.no_grad():
        _ = model(x)  # in case of ActNorm
    state_dict = torch.load(weight_path)
    net_dict = state_dict["net"]
    if opt.fix_statedict:
        net_dict = fix_st(net_dict)
    model.load_state_dict(net_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    accs = dict()
    with tqdm(total=len(opt.corruptions), ncols=80) as pbar:
        for ci, cname in enumerate(opt.corruptions):
            # load dataset
            if cname == 'natural':
                dataset = datasets.CIFAR10(
                    os.path.join(opt.data_root, 'cifar10'),
                    train=False, transform=transform, download=True,
                )
            else:
                dataset = CIFAR10C(
                    os.path.join(opt.data_root, 'CIFAR-10-C'),
                    cname, transform=transform, severity=5
                )
            loader = DataLoader(dataset, batch_size=opt.batch_size,
                                shuffle=False, num_workers=4)
            
            acc_meter = AverageMeter()
            with torch.no_grad():
                for itr, (x, y) in enumerate(loader):
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, dtype=torch.int64, non_blocking=True)

                    z = model(x)
                    loss = F.cross_entropy(z, y)
                    acc, _ = accuracy(z, y, topk=(1, 5))
                    acc_meter.update(acc.item())

            accs[f'{cname}'] = acc_meter.avg

            pbar.set_postfix_str(f'{cname}: {acc_meter.avg:.2f}')
            pbar.update()
    
    avg = np.mean(list(accs.values()))
    accs['avg'] = avg

    pprint.pprint(accs)
    save_name = get_fname(weight_path)
    create_barplot(
        accs, save_name + f' / avg={avg:.2f}',
        os.path.join(opt.fig_dir, save_name+'.png')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--c', type=float, default=0,
                        help='Lipschitz constant: 0 for no SN, positive for soft, negative '
                             'for hard')
    parser.add_argument('--norm_layer', default='batchnorm',
                        help='norm layer to use for constrained nets: batchnorm or actnorm')
    parser.add_argument('--fix_statedict', action='store_true', default=False)
    parser.add_argument('--mod', action='store_true', default=False,
                        help='use increased sensitivity: average pooling shortcut and leaky relu')
    parser.add_argument(
        '--weight_dir',
        type=str,
        help='path to the directory containing model weights',
    )
    parser.add_argument(
        '--weight_path',
        type=str,
        help='path to the  model weights',
    )
    parser.add_argument(
        '--fig_dir',
        type=str, default='/code/figs',
        help='path to the directory saving output figure',
    )
    parser.add_argument(
        '--data_root',
        type=str, default='/data',
        help='root path to cifar10-c directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int, default=1024,
        help='batch size',
    )
    parser.add_argument(
        '--corruptions',
        type=str, nargs='*',
        default=CORRUPTIONS,
        help='testing corruption types',
    )
    parser.add_argument(
        '--gpu_id',
        type=str, default=0,
        help='gpu id to use'
    )

    opt = parser.parse_args()

    if opt.weight_path is not None:
        main(opt, opt.weight_path)
    elif opt.weight_dir is not None:
        for path in glob(f'./{opt.weight_dir}/*.pth'):
            print('\n', path)
            main(opt, path)
    else:
        raise ValueError("Please specify weight_path or weight_dir option.")