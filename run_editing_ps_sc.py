#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: run_editing_ps_sc.py
# --- Creation Date: 30-05-2020
# --- Last Modified: Tue 16 Mar 2021 16:57:15 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Image editing script for VC2 model.
"""

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import os

import pretrained_networks
from training import misc
import collections
from PIL import Image

#----------------------------------------------------------------------------
def image_to_ready(filename):
    image = np.array(Image.open(filename).convert('RGB'))
    image = np.transpose(image, (2, 0, 1))
    image = np.reshape(image, [1]+list(image.shape))
    image = misc.adjust_dynamic_range(image, [0, 255], [-1, 1])
    return image

def image_to_out(image):
    image = misc.adjust_dynamic_range(image, [-1, 1], [0, 255])
    image = np.transpose(image, [0, 2, 3, 1])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    return image

def images_editing(network_pkl, exist_imgs_dir, attr_source_dict, face_source_ls,
                  attr2idx_dict, create_new_G, new_func_name):
    '''
    attr_source_dict: collections.OrderedDict, {'img1.png': ['azimuth', 'haircolor'], 
        'img2.png': ['smile'], ...}
    face_source_ls: list, ['img3.png', 'img4.png', ...]
    attr2idx_dict: {'azimuth': 10, 'haircolor': 17, 'smile': 6, ...}
    '''
    tflib.init_tf()
    print('attr_source_dict:', attr_source_dict)
    print('Loading networks from "%s"...' % network_pkl)
    # _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    _G, _D, I, Gs = misc.load_pkl(network_pkl)
    if create_new_G:
        Gs = Gs.convert(new_func_name=new_func_name)
    # noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    # Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = True

    attr_source_saved = False
    ori_imgs = []
    attr_source_imgs = []
    all_alt_imgs = []
    for face_img in face_source_ls:
        alt_imgs = []
        face_file = os.path.join(exist_imgs_dir, face_img)
        image = image_to_ready(face_file)
        attr_ori = I.run(image)
        # print('attr_ori.shape: ', attr_ori.shape)
        # print('Gs.num_inputs', Gs.num_inputs)
        # print('z.shape:', z.shape)
        ori_img, _ = Gs.run(attr_ori, None, **Gs_kwargs)
        ori_imgs.append(ori_img)
        for k in attr_source_dict:
            attr = attr_ori.copy()
            source_attr_file = os.path.join(exist_imgs_dir, k)
            source_attr_image = image_to_ready(source_attr_file)
            attr_source = I.run(source_attr_image)
            for attr_to_change in attr_source_dict[k]:
                attr[:, attr2idx_dict[attr_to_change]] = attr_source[:, attr2idx_dict[attr_to_change]]
            if not attr_source_saved:
                attr_source_img, _ = Gs.run(attr_source, None, **Gs_kwargs)
                attr_source_imgs.append(attr_source_img)
            alt_img, _ = Gs.run(attr, None, **Gs_kwargs)
            # print('alt_img.shape:', alt_img.shape)
            alt_imgs.append(alt_img)
        attr_source_saved = True
        all_alt_imgs.append(np.concatenate(tuple(alt_imgs), axis=2)) # along height
    ori_imgs = np.concatenate(tuple(ori_imgs), axis=3) # along width
    attr_source_imgs = np.concatenate(tuple(attr_source_imgs), axis=2) # along height
    all_alt_imgs = np.concatenate(tuple(all_alt_imgs), axis=3) # along width
    # Output
    ori_imgs_out = image_to_out(ori_imgs)
    attr_source_imgs_out = image_to_out(attr_source_imgs)
    all_alt_imgs_out = image_to_out(all_alt_imgs)
    PIL.Image.fromarray(ori_imgs_out[0], 'RGB').save(dnnlib.make_run_dir_path('ori_imgs.png'))
    PIL.Image.fromarray(attr_source_imgs_out[0], 'RGB').save(dnnlib.make_run_dir_path('attr_source_imgs.png'))
    PIL.Image.fromarray(all_alt_imgs_out[0], 'RGB').save(dnnlib.make_run_dir_path('all_alt_imgs.png'))

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

def _str_to_attrsourcedict(v):
    v_values = v.strip()[1:-1]
    items = [x.strip() for x in v_values.split(';')]
    attrs = collections.OrderedDict()
    for item in items:
        k, attr_str = item.split(':')[0].strip(), item.split(':')[1].strip()
        attr_str = attr_str[1:-1]
        attr_ls = [x.strip() for x in attr_str.split(',')]
        attrs[k] = attr_ls
    return attrs

def _str_to_attr2idx(v):
    v_values = v.strip()[1:-1]
    items = [x.strip() for x in v_values.split(',')]
    attr2idx_dict = collections.OrderedDict()
    for item in items:
        k, idx = item.split(':')[0].strip(), int(item.split(':')[1].strip())
        attr2idx_dict[k] = idx
    return attr2idx_dict

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='''PS_SC Editing.''')

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_images_editing = subparsers.add_parser('images-editing', help='Images editing')
    parser_images_editing.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_images_editing.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser_images_editing.add_argument('--exist_imgs_dir', help='Dir for used input images', default='inputs', metavar='EXIST')
    parser_images_editing.add_argument('--face_source_ls', help='Image names to provide faces',
                                       default='[img1.png, img2.png]', type=_str_to_list)
    parser_images_editing.add_argument('--attr_source_dict', help='Image names and attrs',
                                       default='{img1.png: [azimuth, haircolor]; img2.png: [smile]}', type=_str_to_attrsourcedict)
    parser_images_editing.add_argument('--attr2idx_dict', help='Attr names to attr idx in latent codes',
                                       default='{azimuth: 10, haircolor: 17, smile: 6}', type=_str_to_attr2idx)
    parser_images_editing.add_argument('--create_new_G', help='If create a new G for projection.', default=False, type=_str_to_bool)
    parser_images_editing.add_argument('--new_func_name', help='new G func name if create new G', default='training.ps_sc_networks2.G_main_ps_sc')

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
        'images-editing': 'run_editing_ps_sc.images_editing',
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
