import random
import argparse, os, yaml
import torch
import torchvision.utils as tvu
import numpy as np
from skimage.metrics import structural_similarity as ssim
from guided_diffusion.unet import create_model
import lpips
from tqdm import tqdm
from datasets import get_dataset, data_transform, inverse_data_transform
from guided_diffusion.unet_ffhq import create_model as create_model_ffhq
from algos.unconditional_ddim import Unconditional

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def init_model(opt, config, model_config, device):
    if config.model_type == 'ffhq':
        model = create_model_ffhq(**model_config)
        model = model.to(device)
        model.eval()

    elif config.model_type == 'imagenet':
        model = create_model(**model_config)
        model = model.to(device)
        model.eval()

    return model

def init_algo(opt, model, H_funcs=None, sigma_y=0.01, deg=None, betas=None):
    algo = Unconditional(model, H_funcs, sigma_y)
    return algo

def prepare_measurement(opt, config, device):
    ## get degradation matrix ##
    deg = opt.deg
    H_funcs = None
    if 'sr' in deg:
        if deg[:10] == 'sr_bicubic':
            factor = int(deg[10:])
            from obs_functions.Hfuncs import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(device)
            H_funcs = SRConv(kernel / kernel.sum(), \
                            config.data.channels, config.data.image_size, device, stride = factor)
        else:
            # Super-Resolution
            blur_by = int(deg[2:])
            from obs_functions.Hfuncs import SuperResolution
            H_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, device)
    elif 'inp' in deg:
        if 'box' in deg:
            missing = torch.zeros([config.data.image_size, config.data.image_size, config.data.channels])
            left = random.randint(16, 112)
            up = random.randint(16, 112)
            #left = 64
            #up = 64
            missing[left:left+128, up:up+128, :] = 1.0
            missing = missing.view(-1).to(device).long()
            missing = torch.nonzero(missing).squeeze() 
        else:
            # Random inpainting
            missing_r = 3 * torch.randperm(config.data.image_size**2)[:int(config.data.image_size**2 * 0.92)].to(device).long()
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        from obs_functions.Hfuncs import Inpainting
        H_funcs = Inpainting(config.data.channels, config.data.image_size, missing, device)
    elif 'deblur_gauss' in deg:
        # Gaussian Deblurring
        from obs_functions.Hfuncs import Deblurring
        sigma = 1.0
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
        H_funcs = Deblurring(kernel / kernel.sum(), config.data.channels, config.data.image_size, device)
    elif 'phase' in deg:
        # Phase Retrieval
        from obs_functions.Hfuncs import PhaseRetrievalOperator
        H_funcs = PhaseRetrievalOperator(oversample=2.0, device=device)
    elif 'hdr' in deg:
        # HDR
        from obs_functions.Hfuncs import HDR
        H_funcs = HDR()   
    elif 'cs' in deg:
        compress_by = int(deg[2:])
        from obs_functions.Hfuncs import WalshHadamardCS
        H_funcs = WalshHadamardCS(config.data.channels, config.data.image_size, compress_by, torch.randperm(config.data.image_size**2, device=device), device)
    elif deg == 'deblur_aniso':
        from obs_functions.Hfuncs import Deblurring2D
        sigma = 20
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
        sigma = 1
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
        H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), config.data.channels, config.data.image_size, device)
    elif deg == 'deblur_nonlinear':
        from obs_functions.Hfuncs import NonlinearBlurOperator
        H_funcs = NonlinearBlurOperator(device, opt_yml_path='./bkse/options/generate_blur/default.yml')
    elif deg == 'color':
        from obs_functions.Hfuncs import Colorization
        H_funcs = Colorization(config.data.image_size, device)
    else:
        print("ERROR: degradation type not supported")
        quit()

    # for linear observations
    # if 'sr' in deg or 'inp' in deg or 'deblur_gauss' in deg:
    opt.sigma_y = 2 * opt.sigma_y #to account for scaling to [-1,1]
    sigma_y = opt.sigma_y

    return H_funcs, sigma_y, deg


def sample_image(opt, config=None, model_config=None, device='cuda'):
    H_funcs, sigma_y, deg = prepare_measurement(opt, config, device)
    model = init_model(opt, config, model_config, device)
    betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

    algo = init_algo(opt, model, H_funcs, sigma_y, deg, betas)

    #get original images and corrupted y_0
    dataset, test_dataset = get_dataset(opt, config)
    
    if opt.subset_start >= 0 and opt.subset_end > 0:
        assert opt.subset_end > opt.subset_start
        test_dataset = torch.utils.data.Subset(test_dataset, range(opt.subset_start, opt.subset_end))
    else:
        opt.subset_start = 0
        opt.subset_end = len(test_dataset)

    print(f'Dataset has size {len(test_dataset)}')    
    
    def seed_worker(worker_id):
        worker_seed = opt.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(opt.seed)
    if 'phase' in opt.deg:
        if config.sampling.batch_size > 1:
            key = input('Recommend using batch size 1. Current batch size is {}, switch to 1? [y/n]'.format(config.sampling.batch_size))
            if key == 'y':
                config.sampling.batch_size = 1
                print('switch to 1')
            else:
                print('keep using {}'.format(config.sampling.batch_size))

    val_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

    # T = 750 -> 0
    skip = 750 // opt.timesteps
    seq = [skip * i for i in range(1, opt.timesteps+1)]
    seq_next = [-1] + list(seq[:-1])

    print(f'Start from {opt.subset_start}')
    idx_init = opt.subset_start
    idx_so_far = opt.subset_start
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    pbar = tqdm(val_loader)
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    # randomized measurement noise
    p_impulse_list = torch.rand(len(pbar)) * 0.2
    sigma_speckle_list = torch.rand(len(pbar)) * 0.4
    
    for i_img, (x_orig, classes) in enumerate(pbar):

        x_orig = x_orig.to(device)
        x_orig = data_transform(config, x_orig)

        # Transform under forward operator
        y_0 = H_funcs.H(x_orig).detach()

        # Add measurement noise
        if opt.noise_type == "gaussian":
            y_0 = y_0 + sigma_y * torch.randn_like(y_0)
        elif opt.noise_type == "impulse":
            # impulse prob
            p = p_impulse_list[i_img]
            # draw random uniforms same shape as img
            rand = torch.rand_like(y_0)
            y_0[rand < p/2] = -1
            y_0[rand > 1-p/2] = 1
        elif opt.noise_type == "speckle":
            y_0 = y_0 * (1 + torch.randn_like(y_0) * sigma_speckle_list[i_img])

        y_pinv = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels, config.data.image_size, config.data.image_size)
        os.makedirs(opt.image_folder, exist_ok=True)

        for i in range(len(y_0)):
            tvu.save_image(
                inverse_data_transform(config, y_pinv[i]), os.path.join(opt.image_folder, f"y0_{idx_so_far + i}.png")
            )

        x = torch.randn(
                        y_0.shape[0],
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=device,
                    )
        
        xt = x
        n = x.shape[0]
        
        b = torch.from_numpy(betas).float().to(device)
        
        xt = nhmc(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, x_orig)
        
        with torch.no_grad():
            final = torch.stack([inverse_data_transform(config, y) for y in xt])

            metrics_sum = [[], [], []]
            for j in range(len(final)):
                if j == len(final) - 1:
                    tvu.save_image(
                        final[j], os.path.join(opt.image_folder, f"{idx_so_far}.png")
                    )
                orig = inverse_data_transform(config, x_orig[0])
                mse = torch.mean((final[j].to(device) - orig) ** 2)
                PSNR = 10 * torch.log10(1 / mse)
                SSIM = ssim(final[j].detach().cpu().numpy(), orig.detach().cpu().numpy(), data_range=final[j].detach().cpu().numpy().max() - final[j].detach().cpu().numpy().min(), channel_axis=0)
                LPIPS = loss_fn_vgg(2*orig-1.0, 2*torch.tensor(final[j]).to(torch.float32).cuda()-1.0)[0,0,0,0]
                metrics_sum[0].append(PSNR.item())
                metrics_sum[1].append(SSIM)
                metrics_sum[2].append(LPIPS.item())

            avg_psnr += np.mean(metrics_sum[0])
            avg_ssim += np.mean(metrics_sum[1])
            avg_lpips += np.mean(metrics_sum[2])

            idx_so_far += y_0.shape[0]
            num_idx = idx_so_far - idx_init

            pbar.set_description("PSNR:{:.4f}, SSIM:{:.5f}, LPIPS:{:.5f}".format(avg_psnr / num_idx, avg_ssim / num_idx, avg_lpips / num_idx))

        del xt, final
        torch.cuda.empty_cache()

    avg_psnr = avg_psnr / num_idx
    avg_ssim = avg_ssim / num_idx
    avg_lpips = avg_lpips / num_idx
    print("Total Average PSNR: {:.3f}".format(avg_psnr))
    print("Total Average SSIM: {:.5f}".format(avg_ssim))
    print("Total Average LPIPS: {:.5f}".format(avg_lpips))
    print("Number of samples: {}".format(num_idx))

    return avg_psnr, avg_ssim, avg_lpips


def nhmc(x, n, b, seq, seq_next, algo, opt, y_0, H_funcs, x_orig):

    orig_pic = []
    for j in range(len(x_orig)):
        orig_pic.append(inverse_data_transform(config, x_orig[j]))

    x = x.detach().requires_grad_()
    epsilon = opt.epsilon
    L = opt.L
    if 'phase_retrieval' in opt.deg:
        warm_up1 = 50
        warm_up2 = 30
        sampling = 1
        gamma_reject = 0.95
        total_epochs = warm_up1 + warm_up2 + sampling
    else:
        warm_up1 = 10
        warm_up2 = 70
        sampling = 1
        gamma_reject = 0.95
        total_epochs = warm_up1 + warm_up2 + sampling

    M_diag = torch.ones_like(x).reshape(-1)
    std_diag = torch.sqrt(M_diag)
    inv_M_diag = 1.0 / M_diag

    final_img_list = []

    d_y = y_0.view(-1).numel()

    rejected = 0
    epoch = 0
    while epoch < total_epochs:

        # Reduce step size after warm up
        if epoch == warm_up1 and epsilon > 0.02:
            epsilon = 0.01

        # initialize momentum
        p = torch.randn_like(x, device=device) * std_diag.view_as(x)
       
        xt = iterative_sampling(x, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)
        error = torch.sum((y_0 - H_funcs.H(xt))**2)
        if opt.unknown_noise:
            if epoch < warm_up1:
                if 'phase_retrieval' in opt.deg:
                    sigma_y = 1.0 + 20.0 * (1 - epoch / warm_up1)**0.5
                else:
                    sigma_y = 0.5 + 2.0 * (1 - epoch / warm_up1)
                loss = (1/(2 * sigma_y**2)) * error
            else:
                loss = (0.5 * d_y) * torch.log(0.5 * error)
        else:
            if 'phase_retrieval' in opt.deg:
                sigma_y = opt.sigma_y + 20.0 * (1 - epoch / (warm_up1+warm_up2))
            else:
                sigma_y = opt.sigma_y + 3.0 * (1 - epoch / (warm_up1+warm_up2))**2
            
            loss = (1/(2 * sigma_y**2)) * error

        current_loss = loss.detach()

        loss_grad = torch.autograd.grad(loss, x, retain_graph=False)[0]
        total_current_loss = (1/2) * torch.sum(x.detach()**2, dim=(1, 2, 3)) + current_loss

        H = total_current_loss + (1/2) * torch.sum(inv_M_diag.view_as(x) * p**2)

        x_proposal = x.detach().clone().requires_grad_(True)

        for l in range(L):

            # update momentum
            p = p - (epsilon / 2) * (x_proposal.detach() + loss_grad)

            x_proposal = x_proposal + epsilon * inv_M_diag.view_as(x) * p 
            x_proposal = x_proposal.detach().requires_grad_(True)

            xt = iterative_sampling(x_proposal, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=True).clip(-1, 1)

            error = torch.sum((y_0 - H_funcs.H(xt))**2)
            if opt.unknown_noise:
                if epoch < warm_up1:
                    loss = (1/(2 * sigma_y**2)) * error
                else:
                    loss = (0.5 * d_y) * torch.log(0.5 * error)
            else:
                loss = (1/(2 * sigma_y**2)) * error
            loss_grad = torch.autograd.grad(loss, x_proposal, retain_graph=False)[0]

            p = p - (epsilon / 2) * (x_proposal.detach() + loss_grad)

        proposal_loss = loss.detach()
        total_proposal_loss = (1/2) * torch.sum(x_proposal.detach()**2, dim=(1, 2, 3)) +  proposal_loss
        H_proposal = total_proposal_loss + (1/2) * torch.sum(inv_M_diag.view_as(x) * p**2)
        delta_H = H_proposal - H

        acceptance_ratio = min(torch.tensor([1], device=device), torch.exp(-delta_H))
        # Heuristic acceptance during warm up
        if epoch < warm_up1 + warm_up2 and total_proposal_loss < total_current_loss:
            accept = True
        else:
            accept = torch.rand(1).item() < acceptance_ratio.item()

        if accept:
            rejected = 0
            epoch += 1

            x_accept = xt.detach().clone()

            # Save sampled images
            if epoch >= warm_up1 + warm_up2:
                final_img_list.append(x_accept[0])
            
            x = x_proposal.detach().clone().requires_grad_(True)
            
            if opt.verbose:
                x_save = [inverse_data_transform(config, y) for y in xt.detach()]
                for j in range(len(x_save)):
                    mse = torch.mean((x_save[j] - orig_pic[j]) ** 2)
                    psnr = 10 * torch.log10(1 / mse)
                    print('epoch', epoch, 'PSNR:', psnr.item())
        else:
            rejected += 1
            if rejected >= 5:
                epoch += 1
                rejected = 0
                break

            # Reduce step size if proposal is rejected
            if epoch < warm_up1 + warm_up2:
                epsilon = epsilon * gamma_reject

            if opt.verbose: print(rejected, 'rejected')
            

    if len(final_img_list) == 0:
        final_img_list.append(x_accept[0])

    return torch.stack(final_img_list)


def iterative_sampling(xt, n, b, seq, seq_next, algo, opt, y_0, tqdm_disable=False):

    for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), disable=tqdm_disable):
        t = (torch.ones(n) * i).to(xt.device)
        next_t = (torch.ones(n) * j).to(xt.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        x0_t, add_up = algo.cal_x0(xt, t, at, at_next)

        xt_next = algo.map_back(x0_t, y_0, add_up, at_next, at)
        xt = xt_next

    return xt

def gaussian_log_prob(x_from, x_to, grad_from, epsilon):
    mean = x_from - epsilon * grad_from
    diff = x_to - mean
    return -(1/4) * (diff.pow(2).sum() / epsilon)

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=5678, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="?",
        help="dataset",
        default="celeba"
    )
    parser.add_argument(
        "--deg", type=str, required=True, help="Degradation Type"
    )
    parser.add_argument(
        "--noise_type", type=str, default="gaussian", help="Type of Measurement Noise"
    )
    parser.add_argument(
        "--unknown_noise",
        action="store_true",
        help="True: use noise-adaptive algorithm",
        default=False
    )
    parser.add_argument(
        "--sigma_y", type=float, required=True, help="True Measurement Noise"
    )
    parser.add_argument(
        "--L", type=float, default=20, help="L for HMC"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.05, help="Epsilon for HMC"
    )
    parser.add_argument(
        "--m", type=float, default=1.0, help="Mass for HMC"
    )

    parser.add_argument(
        "--num_timesteps",
        type=int,
        nargs="?",
        help="Maximum timestep for beta schedule",
        default=1000
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="?",
        help="Number of timesteps for actual sampling",
        default=2
    )
    parser.add_argument(
        '--subset_start', type=int, default=-1
    )
    parser.add_argument(
        '--subset_end', type=int, default=-1
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="exp/samples/ffhq/00000",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Whether to print detailed information during sampling"
    )
    return parser

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

if __name__ == "__main__":
    # Load configurations
    parser = get_parser()
    device = torch.device("cuda")
    opt, unknown = parser.parse_known_args()
    opt.config = 'configs/config_{}.yml'.format(opt.dataset)
    with open(opt.config, "r") as f:
        config = yaml.safe_load(f)
    model_config = config['model']
    config = dict2namespace(config)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
   
    sample_image(opt=opt, config=config, model_config=model_config, device=device)