from load_audface import load_audface_data
import os
import sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from natsort import natsorted
from run_nerf_helpers import *

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir, os.pardir)))

from AudioConditionModel import AudioConditionModel
from LandmarkModels import LandmarkEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, aud_para, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)
    aud = aud_para.unsqueeze(0).expand(inputs_flat.shape[0], -1)
    embedded = torch.cat((embedded, aud), -1)
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, bc_rgb, aud_para, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], bc_rgb[i:i+chunk],
                          aud_para, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render_dynamic_face(H, W, focal, cx, cy, chunk=1024*32, rays=None, bc_rgb=None, aud_para=None,
                        c2w=None, ndc=True, near=0., far=1.,
                        use_viewdirs=False, c2w_staticcam=None,
                        **kwargs):
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w, cx, cy)
        bc_rgb = bc_rgb.reshape(-1, 3)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam, cx, cy)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * \
        torch.ones_like(rays_d[..., :1]), far * \
        torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, bc_rgb, aud_para, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'last_weight']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render(H, W, focal, cx, cy, chunk=1024*32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w, cx, cy)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam, cx, cy)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * \
        torch.ones_like(rays_d[..., :1]), far * \
        torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, aud_paras, bc_img, hwfcxy,
                chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal, cx, cy = hwfcxy

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    last_weights = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, last_weight, _ = render_dynamic_face(
            H, W, focal, cx, cy, chunk=chunk, c2w=c2w[:3,
                                                      :4], aud_para=aud_paras[i], bc_rgb=bc_img,
            **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        last_weights.append(last_weight.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    last_weights = np.stack(last_weights, 0)

    return rgbs, disps, last_weights


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = FaceNeRF(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, dim_aud=args.dim_aud,
                     output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = FaceNeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                              input_ch=input_ch, dim_aud=args.dim_aud,
                              output_ch=output_ch, skips=skips,
                              input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    def network_query_fn(inputs, viewdirs, aud_para, network_fn): \
        return run_network(inputs, viewdirs, aud_para, network_fn,
                           embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(
        params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in natsorted(
            os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    learned_codes_dict = None
    AudNet_state = None
    AudAttNet_state = None
    optimizer_aud_state = None
    optimizer_audatt_state = None
    aud_cond_state = None
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        AudNet_state = ckpt['network_audnet_state_dict']
        optimizer_aud_state = ckpt['optimizer_aud_state_dict']

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if 'network_audattnet_state_dict' in ckpt:
            AudAttNet_state = ckpt['network_audattnet_state_dict']
        if 'optimize_audatt_state_dict' in ckpt:
            optimizer_audatt_state = ckpt['optimize_audatt_state_dict']

    if args.load_aud_cond:
        aud_cond_state = torch.load(args.load_aud_cond)
    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, learned_codes_dict, \
        AudNet_state, optimizer_aud_state, AudAttNet_state, optimizer_audatt_state, aud_cond_state


def raw2outputs(raw, z_vals, rays_d, bc_rgb, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
        torch.exp(-(act_fn(raw)+1e-6)*dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(
        dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    rgb = torch.cat((rgb[:, :-1, :], bc_rgb.unsqueeze(1)), dim=1)
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * \
        torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map),
                            depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                bc_rgb,
                aud_para,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)
        t_rand[..., -1] = 1.0
        z_vals = lower + (upper - lower) * t_rand
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]
    raw = network_query_fn(pts, viewdirs, aud_para, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, bc_rgb, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, aud_para, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, bc_rgb, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ret['last_weight'] = weights[..., -1]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=2048,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_false',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--N_iters", type=int, default=400000,
                        help='number of iterations')
    # args.load_aud_cond
    parser.add_argument("--load_aud_cond", type=str, default=None,
                        help='load from audcond model')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_false',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='audface',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_false',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # face flags
    parser.add_argument("--with_test", type=int, default=0,
                        help='whether to use test set')
    parser.add_argument("--dim_aud", type=int, default=64,
                        help='dimension of audio features for NeRF')
    parser.add_argument("--sample_rate", type=float, default=0.95,
                        help="sample rate in a bounding box")
    parser.add_argument("--near", type=float, default=0.3,
                        help="near sampling plane")
    parser.add_argument("--far", type=float, default=0.9,
                        help="far sampling plane")
    parser.add_argument("--test_file", type=str, default='transforms_test.json',
                        help='test file')
    parser.add_argument("--test_rof_file", type=str, default='rof_mean.tar',
                        help='test rest-of-face embedding mean saved as npy')
    parser.add_argument("--aud_file", type=str, default='aud.npy',
                        help='test audio deepspeech file')
    parser.add_argument("--win_size", type=int, default=16,
                        help="windows size of audio feature")
    parser.add_argument("--smo_size", type=int, default=8,
                        help="window size for smoothing audio features")
    parser.add_argument('--nosmo_iters', type=int, default=200000,
                        help='number of iterations befor applying smoothing on audio features')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=10000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data

    if args.dataset_type == 'audface':
        if args.with_test == 1:
            poses, auds, bc_img, hwfcxy, rof_emb = \
                load_audface_data(basedir=args.datadir, testskip=args.testskip,
                                  test_file=args.test_file, test_rof_file=args.test_rof_file, aud_file=args.aud_file)
            images = np.zeros(1)
        else:
            images, poses, auds, lms, bc_img, hwfcxy, sample_rects, mouth_rects, i_split = load_audface_data(
                basedir=args.datadir, testskip=args.testskip)
        print('Loaded audface', images.shape, hwfcxy, args.datadir)
        if args.with_test == 0:
            i_train, i_val = i_split

        near = args.near
        far = args.far
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal, cx, cy = hwfcxy
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    hwfcxy = [H, W, focal, cx, cy]

    # if args.render_test:
    #     render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, \
        learned_codes, AudNet_state, optimizer_aud_state, AudAttNet_state, optimizer_audatt_state, aud_cond_state \
        = create_nerf(args)
    global_step = start

    AudNet = AudioNet(args.dim_aud, args.win_size).to(device)
    AudAttNet = AudioAttNet().to(device)
    aud_cond_model = AudioConditionModel().to(device)
    landmark_encoder = LandmarkEncoder().to(device)
    optimizer_Aud = torch.optim.Adam(
        params=list(AudNet.parameters()), lr=args.lrate, betas=(0.9, 0.999))
    optimizer_AudAtt = torch.optim.Adam(
        params=list(AudAttNet.parameters()), lr=args.lrate, betas=(0.9, 0.999))

    if AudNet_state is not None:
        AudNet.load_state_dict(AudNet_state, strict=False)
    if optimizer_aud_state is not None:
        optimizer_Aud.load_state_dict(optimizer_aud_state)
    if AudAttNet_state is not None:
        AudAttNet.load_state_dict(AudAttNet_state, strict=False)
    if optimizer_audatt_state is not None:
        optimizer_AudAtt.load_state_dict(optimizer_audatt_state)
    if aud_cond_state is not None:
        aud_cond_model.load_state_dict(aud_cond_state)
        landmark_encoder.load_state_dict(aud_cond_model.landmark_encoder.state_dict())
        AudNet.load_state_dict(aud_cond_model.audionet.state_dict())
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move training data to GPU
    bc_img = torch.Tensor(bc_img).to(device).float()/255.0
    poses = torch.Tensor(poses).to(device).float()
    auds = torch.Tensor(auds).to(device).float()
    rof_count = 0
    rof_mean = torch.zeros(1, 64)
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            # Default is smoother render_poses path
            images = None
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
                'test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses.shape)
            auds_val = AudNet(auds)

            rof_embs = torch.cat([rof_emb]*auds.size(0), 0)
            auds_val = torch.cat([auds_val, rof_embs], 1)

            rgbs, disp, last_weight = render_path(poses, auds_val, bc_img, hwfcxy, args.chunk, render_kwargs_test,
                                                  gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            np.save(os.path.join(testsavedir, 'last_weight.npy'), last_weight)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(
                testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            return

    num_frames = images.shape[0]


    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    print('N_rand', N_rand, 'no_batching',
          args.no_batching, 'sample_rate', args.sample_rate)
    use_batching = not args.no_batching

    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p, cx, cy)
                         for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i]
                             for i in i_train], 0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('VAL views are', i_val)

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = torch.as_tensor(imageio.imread(
                images[img_i])).to(device).float()/255.0
            pose = poses[img_i, :3, :4]
            lm = torch.as_tensor(np.loadtxt(lms[img_i]).astype(np.float32)).to(device)
            lm = lm/target.size(0)
            rect = sample_rects[img_i]
            mouth_rect = mouth_rects[img_i]
            aud = auds[img_i]
            rof_emb, mouth_emb = landmark_encoder(lm)
            rof_mean = (rof_count*rof_mean)+rof_emb
            rof_count += 1
            rof_mean /= rof_count
            if global_step >= args.nosmo_iters:
                smo_half_win = int(args.smo_size / 2)
                left_i = img_i - smo_half_win
                right_i = img_i + smo_half_win
                pad_left, pad_right = 0, 0
                if left_i < 0:
                    pad_left = -left_i
                    left_i = 0
                if right_i > i_train.shape[0]:
                    pad_right = right_i-i_train.shape[0]
                    right_i = i_train.shape[0]
                auds_win = auds[left_i:right_i]
                if pad_left > 0:
                    auds_win = torch.cat(
                        (torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
                if pad_right > 0:
                    auds_win = torch.cat(
                        (auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
                auds_win = AudNet(auds_win)
                aud = auds_win[smo_half_win]
                aud_smo = AudAttNet(auds_win)
                aud_smo = torch.cat([aud_smo, torch.squeeze(rof_emb)], 0)
            else:
                aud = AudNet(aud.unsqueeze(0))
                aud = torch.cat([aud, torch.squeeze(rof_emb)], 0)

            if N_rand is not None:
                rays_o, rays_d = get_rays(
                    H, W, focal, torch.Tensor(pose), cx, cy)  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH),
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(
                        0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                if args.sample_rate > 0:
                    rect_inds = (coords[:, 0] >= rect[0]) & (
                        coords[:, 0] <= rect[0] + rect[2]) & (
                            coords[:, 1] >= rect[1]) & (
                                coords[:, 1] <= rect[1] + rect[3])
                    coords_rect = coords[rect_inds]
                    coords_norect = coords[~rect_inds]
                    rect_num = int(N_rand*args.sample_rate)
                    norect_num = N_rand - rect_num
                    select_inds_rect = np.random.choice(
                        coords_rect.shape[0], size=[rect_num], replace=False)  # (N_rand,)
                    # (N_rand, 2)
                    select_coords_rect = coords_rect[select_inds_rect].long()
                    select_inds_norect = np.random.choice(
                        coords_norect.shape[0], size=[norect_num], replace=False)  # (N_rand,)
                    # (N_rand, 2)
                    select_coords_norect = coords_norect[select_inds_norect].long(
                    )
                    select_coords = torch.cat(
                        (select_coords_rect, select_coords_norect), dim=0)
                else:
                    select_inds = np.random.choice(
                        coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()

                rays_o = rays_o[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0],
                                  select_coords[:, 1]]  # (N_rand, 3)
                bc_rgb = bc_img[select_coords[:, 0],
                                select_coords[:, 1]]

        #####  Core optimization loop  #####
        if global_step >= args.nosmo_iters:
            rgb, disp, acc, _, extras = render_dynamic_face(H, W, focal, cx, cy, chunk=args.chunk, rays=batch_rays,
                                                            aud_para=aud_smo, bc_rgb=bc_rgb,
                                                            verbose=i < 10, retraw=True,
                                                            **render_kwargs_train)
        else:
            rgb, disp, acc, _, extras = render_dynamic_face(H, W, focal, cx, cy, chunk=args.chunk, rays=batch_rays,
                                                            aud_para=aud, bc_rgb=bc_rgb,
                                                            verbose=i < 10, retraw=True,
                                                            **render_kwargs_train)

        optimizer.zero_grad()
        optimizer_Aud.zero_grad()
        optimizer_AudAtt.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][..., -1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()
        optimizer_Aud.step()
        if global_step >= args.nosmo_iters:
            optimizer_AudAtt.step()
        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        for param_group in optimizer_Aud.param_groups:
            param_group['lr'] = new_lrate

        for param_group in optimizer_AudAtt.param_groups:
            param_group['lr'] = new_lrate*5
        ################################

        dt = time.time()-time0

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}_head.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'network_audnet_state_dict': AudNet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_aud_state_dict': optimizer_Aud.state_dict(),
                'network_audattnet_state_dict': AudAttNet.state_dict(),
                'optimizer_audatt_state_dict': optimizer_AudAtt.state_dict(),
            }, path)
            print('Saved checkpoints at', path)
            rof_path = os.path.join(args.datadir, 'rof_mean.tar')
            torch.save(rof_mean, rof_path)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_val].shape)
            auds_val = AudNet(auds[i_val])
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_val]).to(
                    device), auds_val, bc_img, hwfcxy, args.chunk, render_kwargs_test, gt_imgs=None, savedir=testsavedir)
            print('Saved test set')

        if i % args.i_print == 0:
            tqdm.write(
                f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
