import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

# from models import RDN
from mbarn import MBARN
from utils import convert_rgb_to_y, denormalize, calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default='data/results/x3/epoch_9.pth')
    parser.add_argument('--image-file', type=str, default='data/img_043.png')
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--n_channels', type=int, default=3)
    parser.add_argument('--n_resgroups', type=int, default=1)
    parser.add_argument('--n_resblocks', type=int, default=20)
    parser.add_argument('--n_feats', type=int, default=64)
    parser.add_argument('--reductions', type=int, default=16)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    model = MBARN(scale_factor=args.scale,
                num_channels=args.n_channels,
                n_resgroups=args.n_resgroups,
                n_resblocks=args.n_resblocks,
                n_feats=args.n_feats,
                reduction=args.reductions).to(device)


    model.load_state_dict(torch.load(args.weights_file, map_location=device))

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = hr.resize((hr.width // args.scale, hr.height // args.scale), resample=pil_image.BICUBIC)
    bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
    bicubic.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
    hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
    lr = torch.from_numpy(lr).to(device)
    hr = torch.from_numpy(hr).to(device)
    # print(lr.shape)
    with torch.no_grad():
        preds = model(lr).squeeze(0)

    preds_y = convert_rgb_to_y(denormalize(preds), dim_order='chw')
    hr_y = convert_rgb_to_y(denormalize(hr.squeeze(0)), dim_order='chw')

    preds_y = preds_y[args.scale:-args.scale, args.scale:-args.scale]
    hr_y = hr_y[args.scale:-args.scale, args.scale:-args.scale]

    psnr = calc_psnr(hr_y, preds_y)
    print('PSNR: {:.2f}'.format(psnr))

    output = pil_image.fromarray(denormalize(preds).permute(1, 2, 0).byte().cpu().numpy())
    output.save(args.image_file.replace('.', '_mban_x{}.'.format(args.scale)))
