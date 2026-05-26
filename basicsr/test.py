import logging
import torch
from os import path as osp
import os
import argparse
import yaml

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs, imwrite, tensor2img
from basicsr.utils.options import dict2str, parse_options
from basicsr.utils.calculate import calculate_rmse, calculate_bias, calculate_r2, calculate_psnr_ocean, \
    calculate_ssim_ocean

def calculate_sst_gradient(img_tensor):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img_tensor.dtype, device=img_tensor.device).view(1, 1,
                                                                                                                   3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img_tensor.dtype, device=img_tensor.device).view(1, 1,
                                                                                                                   3, 3)
    padding = (1, 1, 1, 1)
    img_pad = torch.nn.functional.pad(img_tensor, padding, mode='reflect')
    grad_x = torch.nn.functional.conv2d(img_pad, kx)
    grad_y = torch.nn.functional.conv2d(img_pad, ky)
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
    return grad_mag


def test_pipeline(root_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()

    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)

    opt['is_train'] = False
    opt['dist'] = False
    opt['rank'] = 0
    opt['world_size'] = 1
    opt['num_gpu'] = 1

    if 'path' not in opt: opt['path'] = {}
    opt['path']['root'] = root_path
    if 'experiments_root' not in opt['path']:
        opt['path']['experiments_root'] = osp.join(root_path, 'experiments', opt['name'])
    if 'log' not in opt['path']:
        opt['path']['log'] = osp.join(opt['path']['experiments_root'], 'log')
    if 'visualization' not in opt['path']:
        opt['path']['visualization'] = osp.join(opt['path']['experiments_root'], 'visualization')

    nc_save_root = osp.join(opt['path']['experiments_root'], 'nc_results')

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)

    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        test_loaders.append(test_loader)

    ### The source code is currently incomplete and will be fully released once the manuscript is accepted by the journal.

    model = build_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        curr_nc_dir = osp.join(nc_save_root, test_set_name)
        os.makedirs(curr_nc_dir, exist_ok=True)

        rmse_list, bias_list, r2_list, psnr_list, ssim_list = [], [], [], [], []
        d_rmse_list, d_bias_list, d_psnr_list, d_ssim_list = [], [], [], []

        if len(rmse_list) > 0:
            avg_rmse = sum(rmse_list) / len(rmse_list)
            avg_bias = sum(bias_list) / len(bias_list)
            avg_r2 = sum(r2_list) / len(r2_list)
            avg_psnr = sum(psnr_list) / len(psnr_list)
            avg_ssim = sum(ssim_list) / len(ssim_list)

            avg_d_rmse = sum(d_rmse_list) / len(d_rmse_list)
            avg_d_bias = sum(d_bias_list) / len(d_bias_list)
            avg_d_psnr = sum(d_psnr_list) / len(d_psnr_list)
            avg_d_ssim = sum(d_ssim_list) / len(d_ssim_list)

            logger.info(f'------------------------------------------\n'
                        f'Average Results for {test_set_name}:\n'
                        f'RMSE: {avg_rmse:.4f} | R2: {avg_r2:.4f} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | Bias: {avg_bias:.4f}\n'
                        f'Delta SST RMSE: {avg_d_rmse:.4f} | PSNR: {avg_d_psnr:.4f} | SSIM: {avg_d_ssim:.4f} | Bias: {avg_d_bias:.4f}\n'
                        f'------------------------------------------')


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)