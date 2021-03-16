#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: run_generator_ps_sc.py
# --- Creation Date: 26-05-2020
# --- Last Modified: Tue 16 Mar 2021 17:19:08 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Generator run for PS-SC.
"""

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import os
import collections
import cv2

import pretrained_networks
from training import misc
from training.utils import get_grid_latents, get_return_v, add_outline, save_atts
from run_editing_ps_sc import image_to_ready
from run_editing_ps_sc import image_to_out
from PIL import Image, ImageDraw, ImageFont
from metrics.metric_defaults import metric_defaults

#----------------------------------------------------------------------------

def generate_images(network_pkl, seeds, create_new_G, new_func_name):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    # _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    _G, _D, I, Gs = misc.load_pkl(network_pkl)
    if create_new_G:
        Gs = Gs.convert(new_func_name=new_func_name)
    # noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    # Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = True

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        # tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images, _ = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        images = misc.adjust_dynamic_range(images, [-1, 1], [0, 255])
        # np.clip(images, 0, 255, out=images)
        images = np.transpose(images, [0, 2, 3, 1])
        images = np.rint(images).clip(0, 255).astype(np.uint8)
        PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('seed%04d.png' % seed))

def generate_domain_shift(network_pkl, seeds, domain_dim):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    # _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    _G, _D, I, Gs = misc.load_pkl(network_pkl)

    Gs_kwargs = dnnlib.EasyDict()
    # Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = True

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        # tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images_1, _ = Gs.run(z, None, **Gs_kwargs) # [minibatch, c, height, width]
        z[:, domain_dim] = -z[:, domain_dim]
        images_2, _ = Gs.run(z, None, **Gs_kwargs) # [minibatch, c, height, width]
        images = np.concatenate((images_1, images_2), axis=3)
        images = misc.adjust_dynamic_range(images, [-1, 1], [0, 255])
        # np.clip(images, 0, 255, out=images)
        images = np.transpose(images, [0, 2, 3, 1])
        images = np.rint(images).clip(0, 255).astype(np.uint8)
        PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('seed%04d.png' % seed))

def generate_traversals(network_pkl, seeds, tpl_metric, n_samples_per, topk_dims_to_show, return_atts=False, bound=2):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    # _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    _G, _D, I, Gs = get_return_v(misc.load_pkl(network_pkl), 4)
    # noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    # Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = True
    if return_atts:
        Gs_kwargs.return_atts = True,

    n_continuous = Gs.input_shape[1]
    grid_labels = np.zeros([1,0], dtype=np.float32)

    # Eval tpl
    if topk_dims_to_show > 0:
        metric_args = metric_defaults[tpl_metric]
        metric = dnnlib.util.call_func_by_name(**metric_args)
        met_outs = metric._evaluate(Gs, {}, 1)
        if 'tpl_per_dim' in met_outs:
            avg_distance_per_dim = met_outs['tpl_per_dim'] # shape: (n_continuous)
            topk_dims = np.argsort(avg_distance_per_dim)[::-1][:topk_dims_to_show] # shape: (20)
        else:
            topk_dims = np.arange(min(topk_dims_to_show, n_continuous))
    else:
        topk_dims = np.arange(n_continuous)

    for seed_idx, seed in enumerate(seeds):
        grid_size, grid_latents, grid_labels = get_grid_latents(
            0, n_continuous, n_samples_per, Gs, grid_labels, topk_dims)
        # images, _ = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        grid_fakes, atts = get_return_v(Gs.run(grid_latents,
                            grid_labels,
                            is_validation=True,
                            minibatch_size=2,
                            randomize_noise=True), 2)
        if return_atts:
            atts = atts[:, topk_dims]
            save_atts(atts,
                      filename=dnnlib.make_run_dir_path('atts_seed%04d.png' % seed),
                      grid_size=grid_size,
                      drange=[0, 1],
                      grid_fakes=grid_fakes,
                      n_samples_per=n_samples_per)
        grid_fakes = add_outline(grid_fakes, width=1)
        misc.save_image_grid(grid_fakes,
                             dnnlib.make_run_dir_path(
                                 'travs_seed%04d.png' % seed),
                             drange=[-1., 1.],
                             grid_size=grid_size)

def expand_traversal(attr, attr_idx, traversal_frames):
    '''
    attr: [1, n_codes]
    '''
    attrs_trav = np.tile(attr, [traversal_frames]+[1]*(attr.ndim-1))
    attrs_trav[:, attr_idx] = np.linspace(-2., 2., num=traversal_frames)
    return attrs_trav

def single_image_to_out(image):
    image = misc.adjust_dynamic_range(image, [-1, 1], [0, 255])
    image = np.transpose(image, [1, 2, 0])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    return image

def draw_imgs_and_text(semantics_frame_np, used_semantics_ls, img_h, img_w):
    c, h, w = semantics_frame_np.shape
    semantics_frame_np = np.transpose(semantics_frame_np, [1, 2, 0])
    semantics_frame_np = cv2.resize(semantics_frame_np, dsize=(w//4, h//4))
    semantics_frame_np = np.transpose(semantics_frame_np, [2, 0, 1])
    semantics_frame_np = np.concatenate((np.zeros((3, 100, w//4), dtype=semantics_frame_np.dtype)-1,
                                         semantics_frame_np), axis=1)
    semantics_frame = single_image_to_out(semantics_frame_np)
    new_img = Image.fromarray(semantics_frame, 'RGB')
    draw = ImageDraw.Draw(new_img)
    font = ImageFont.truetype("LiberationSans-Regular.ttf", 14)
    for i, semantic_name in enumerate(used_semantics_ls):
        text_w = 5 + (img_w // 4) * i
        text_h = 70
        draw.text((text_w, text_h), semantic_name, font=font, fill=(255, 255, 255))
    return new_img

def generate_gifs(network_pkl, exist_imgs_dir,
                  used_imgs_ls, used_semantics_ls, attr2idx_dict,
                  create_new_G, new_func_name, traversal_frames=20):
    '''
    used_imgs_ls: ['img1.png', 'img2.png', ...]
    used_semantics_ls: ['azimuth', 'haircolor', ...]
    attr2idx_dict: {'azimuth': 10, 'haircolor': 17, 'smile': 6, ...}
    '''
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    # _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    _G, _D, I, Gs = misc.load_pkl(network_pkl)
    if create_new_G:
        Gs = Gs.convert(new_func_name=new_func_name)
    # noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    # Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = True

    ori_imgs = []
    semantics_all_imgs_ls = []
    for in_img_name in used_imgs_ls:
        img_file = os.path.join(exist_imgs_dir, in_img_name)
        image = image_to_ready(img_file)
        attr_ori = I.run(image)
        ori_img, _ = Gs.run(attr_ori, None, **Gs_kwargs)
        _, c, img_h, img_w = ori_img.shape
        ori_imgs.append(ori_img)
        semantics_per_img_ls = []
        for i in range(len(used_semantics_ls)):
            attr = attr_ori.copy()
            attrs_trav = expand_traversal(attr, attr2idx_dict[used_semantics_ls[i]], traversal_frames) # [n_trav, n_codes]
            imgs_trav, _ = Gs.run(attrs_trav, None, **Gs_kwargs) # [n_trav, c, h, w]
            semantics_per_img_ls.append(imgs_trav)
        semantics_per_img_np = np.concatenate(tuple(semantics_per_img_ls), axis=3) # [n_trav, c, h, w*n_used_attrs]
        semantics_all_imgs_ls.append(semantics_per_img_np)
    semantics_all_imgs_np = np.concatenate(tuple(semantics_all_imgs_ls), axis=2) # [n_trav, c, h*n_imgs, w*n_used_attrs]
    imgs_to_save = [draw_imgs_and_text(semantics_all_imgs_np[i], used_semantics_ls, img_h, img_w) for i in range(len(semantics_all_imgs_np))]
    imgs_to_save[0].save(dnnlib.make_run_dir_path('traversals.gif'), format='GIF',
                         append_images=imgs_to_save[1:], save_all=True, duration=100, loop=0)



#----------------------------------------------------------------------------

def style_mixing_example(network_pkl, row_seeds, col_seeds, truncation_psi, col_styles, minibatch_size=4):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg') # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None) # [minibatch, layer, component]
    all_w = w_avg + (all_w - w_avg) * truncation_psi # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    print('Generating images...')
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].copy()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            image_dict[(row_seed, col_seed)] = image

    print('Saving images...')
    for (row_seed, col_seed), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(dnnlib.make_run_dir_path('%d-%d.png' % (row_seed, col_seed)))

    print('Saving image grid...')
    _N, _C, H, W = Gs.output_shape
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([None] + row_seeds):
        for col_idx, col_seed in enumerate([None] + col_seeds):
            if row_seed is None and col_seed is None:
                continue
            key = (row_seed, col_seed)
            if row_seed is None:
                key = (col_seed, col_seed)
            if col_seed is None:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
    canvas.save(dnnlib.make_run_dir_path('grid.png'))

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return range(int(m.group(1)), int(m.group(2))+1)
    vals = s.split(',')
    return [int(x) for x in vals]

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _str_to_list(v):
    v_values = v.strip()[1:-1]
    module_list = [x.strip() for x in v_values.split(',')]
    return module_list

def _str_to_attr2idx(v):
    v_values = v.strip()[1:-1]
    items = [x.strip() for x in v_values.split(',')]
    attr2idx_dict = collections.OrderedDict()
    for item in items:
        k, idx = item.split(':')[0].strip(), int(item.split(':')[1].strip())
        attr2idx_dict[k] = idx
    return attr2idx_dict

#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate ffhq uncurated images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=6600-6625 --truncation-psi=0.5

  # Generate ffhq curated images (matches paper Figure 11)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=66,230,389,1518 --truncation-psi=1.0

  # Generate uncurated car images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=6000-6025 --truncation-psi=0.5

  # Generate style mixing example (matches style mixing video clip)
  python %(prog)s style-mixing-example --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --row-seeds=85,100,75,458,1500 --col-seeds=55,821,1789,293 --truncation-psi=1.0
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 generator.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_generate_images = subparsers.add_parser('generate-images', help='Generate images')
    parser_generate_images.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_images.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', required=True)
    parser_generate_images.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser_generate_images.add_argument('--create_new_G', help='If create a new G for projection.', default=False, type=_str_to_bool)
    parser_generate_images.add_argument('--new_func_name', help='new G func name if create new G', default='training.ps_sc_networks2.G_main_ps_sc')

    parser_style_mixing_example = subparsers.add_parser('style-mixing-example', help='Generate style mixing video')
    parser_style_mixing_example.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_style_mixing_example.add_argument('--row-seeds', type=_parse_num_range, help='Random seeds to use for image rows', required=True)
    parser_style_mixing_example.add_argument('--col-seeds', type=_parse_num_range, help='Random seeds to use for image columns', required=True)
    parser_style_mixing_example.add_argument('--col-styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-6')
    parser_style_mixing_example.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_style_mixing_example.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_generate_travs = subparsers.add_parser('generate-traversals', help='Generate traversals')
    parser_generate_travs.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_travs.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', required=True)
    parser_generate_travs.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser_generate_travs.add_argument('--tpl_metric', help='TPL to use', default='tpl', type=str)
    parser_generate_travs.add_argument('--n_samples_per', help='N samplers per row', default=7, type=int)
    parser_generate_travs.add_argument('--topk_dims_to_show', help='Top k dims to show', default=-1, type=int)
    parser_generate_travs.add_argument('--return_atts', help='If save atts.', default=False, type=_str_to_bool)
    parser_generate_travs.add_argument('--bound', help='Traversal bound', default=2, type=float)

    parser_generate_domain_shift = subparsers.add_parser('generate-domain-shift', help='Generate traversals')
    parser_generate_domain_shift.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_domain_shift.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', required=True)
    parser_generate_domain_shift.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser_generate_domain_shift.add_argument('--domain_dim', help='The dim denoting domain', default=0, type=int)

    parser_generate_gifs = subparsers.add_parser('generate-gifs', help='Generate gifs')
    parser_generate_gifs.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_gifs.add_argument('--exist_imgs_dir', help='Dir for used input images', default='inputs', metavar='EXIST')
    parser_generate_gifs.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser_generate_gifs.add_argument('--used_imgs_ls', help='Image names to use', default='[img1.png, img2.png]', type=_str_to_list)
    parser_generate_gifs.add_argument('--used_semantics_ls', help='Semantics to use', default='[azimuth, haircolor]', type=_str_to_list)
    parser_generate_gifs.add_argument('--attr2idx_dict', help='Attr names to attr idx in latent codes',
                                       default='{azimuth: 10, haircolor: 17, smile: 6}', type=_str_to_attr2idx)
    parser_generate_gifs.add_argument('--create_new_G', help='If create a new G for projection.', default=False, type=_str_to_bool)
    parser_generate_gifs.add_argument('--new_func_name', help='new G func name if create new G', default='training.ps_sc_networks2.G_main_ps_sc')

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = subcmd

    func_name_map = {
        'generate-images': 'run_generator_ps_sc.generate_images',
        'style-mixing-example': 'run_generator_ps_sc.style_mixing_example',
        'generate-traversals': 'run_generator_ps_sc.generate_traversals',
        'generate-domain-shift': 'run_generator_ps_sc.generate_domain_shift',
        'generate-gifs': 'run_generator_ps_sc.generate_gifs'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
