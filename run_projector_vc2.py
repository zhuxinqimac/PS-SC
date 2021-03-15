#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: run_projector_vc2.py
# --- Creation Date: 23-05-2020
# --- Last Modified: Sat 23 May 2020 16:22:41 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Projecting main file for vc2.
"""

import argparse
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import os
import glob

import projector_vc2
import pretrained_networks
from training import dataset
from training import misc
from PIL import Image

#----------------------------------------------------------------------------

def project_image(proj, targets, I_net, png_prefix, num_snapshots):
    snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
    misc.save_image_grid(targets, png_prefix + 'target.png', drange=[-1,1])
    proj.start(targets, I_net)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        if proj.get_cur_step() in snapshot_steps:
            misc.save_image_grid(proj.get_images(), png_prefix + 'step%04d.png' % proj.get_cur_step(), drange=[-1,1])
    print('\r%-30s\r' % '', end='', flush=True)

#----------------------------------------------------------------------------

def project_generated_images(network_pkl, seeds, num_snapshots, create_new_G, new_func_name):
    print('Loading networks from "%s"...' % network_pkl)
    tflib.init_tf()
    # _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    _G, _D, I, Gs = misc.load_pkl(network_pkl)
    proj = projector_vc2.ProjectorVC2()
    proj.set_network(Gs, create_new_G, new_func_name)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.return_atts = True
    Gs_kwargs.randomize_noise = True

    for seed_idx, seed in enumerate(seeds):
        print('Projecting seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
        images, _ = Gs.run(z, None, **Gs_kwargs)
        project_image(proj, targets=images, I_net=I, png_prefix=dnnlib.make_run_dir_path('seed%04d-' % seed), num_snapshots=num_snapshots)

#----------------------------------------------------------------------------

def project_real_dataset_images(network_pkl, dataset_name, data_dir, num_images, num_snapshots, create_new_G, new_func_name):
    print('Loading networks from "%s"...' % network_pkl)
    tflib.init_tf()
    # _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    _G, _D, I, Gs = misc.load_pkl(network_pkl)
    proj = projector_vc2.ProjectorVC2()
    proj.set_network(Gs, create_new_G, new_func_name)

    print('Loading images from "%s"...' % dataset_name)
    dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=0, repeat=False, shuffle_mb=0)
    assert dataset_obj.shape == Gs.output_shape[1:]

    for image_idx in range(num_images):
        print('Projecting image %d/%d ...' % (image_idx, num_images))
        images, _labels = dataset_obj.get_minibatch_np(1)
        images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        project_image(proj, targets=images, I_net=I, png_prefix=dnnlib.make_run_dir_path('image%04d-' % image_idx), num_snapshots=num_snapshots)

#----------------------------------------------------------------------------

def project_real_other_images(network_pkl, data_dir, num_snapshots, create_new_G, new_func_name):
    print('Loading networks from "%s"...' % network_pkl)
    tflib.init_tf()
    # _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    _G, _D, I, Gs = misc.load_pkl(network_pkl)
    proj = projector_vc2.ProjectorVC2()
    proj.set_network(Gs, create_new_G, new_func_name)

    img_paths = glob.glob(os.path.join(data_dir, '*'))
    num_images = len(img_paths)
    for image_idx, img_path in enumerate(img_paths):
        print('Projecting image %d/%d ...' % (image_idx, num_images))
        images = np.array(Image.open(img_path).convert('RGB'))
        images = np.transpose(images, (2, 0, 1))
        images = np.reshape(images, [1]+list(images.shape))
        images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        project_image(proj, targets=images, I_net=I, png_prefix=dnnlib.make_run_dir_path('image%04d-' % image_idx), num_snapshots=num_snapshots)

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

#----------------------------------------------------------------------------

_examples = '''examples:

  # Project generated images
  python %(prog)s project-generated-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=0,1,5

  # Project real images
  python %(prog)s project-real-images --network=gdrive:networks/stylegan2-car-config-f.pkl --dataset=car --data-dir=~/datasets

'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''VC2 projector.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    project_generated_images_parser = subparsers.add_parser('project-generated-images', help='Project generated images')
    project_generated_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_generated_images_parser.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', default=range(3))
    project_generated_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    project_generated_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    project_generated_images_parser.add_argument('--create_new_G', help='If create a new G for projection.', default=False, type=_str_to_bool)
    project_generated_images_parser.add_argument('--new_func_name', help='new G func name if create new G', default='training.vc_networks2.G_main_vc2')

    project_real_dataset_images_parser = subparsers.add_parser('project-real-dataset-images', help='Project real images')
    project_real_dataset_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_real_dataset_images_parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    project_real_dataset_images_parser.add_argument('--dataset', help='Training dataset', dest='dataset_name', required=True)
    project_real_dataset_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    project_real_dataset_images_parser.add_argument('--num-images', type=int, help='Number of images to project (default: %(default)s)', default=3)
    project_real_dataset_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    project_real_dataset_images_parser.add_argument('--create_new_G', help='If create a new G for projection.', default=False, type=_str_to_bool)
    project_real_dataset_images_parser.add_argument('--new_func_name', help='new G func name if create new G', default='training.vc_networks2.G_main_vc2')

    project_real_other_images_parser = subparsers.add_parser('project-real-other-images', help='Project real images')
    project_real_other_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_real_other_images_parser.add_argument('--data-dir', help='Dir of images to project', required=True)
    project_real_other_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    project_real_other_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    project_real_other_images_parser.add_argument('--create_new_G', help='If create a new G for projection.', default=False, type=_str_to_bool)
    project_real_other_images_parser.add_argument('--new_func_name', help='new G func name if create new G', default='training.vc_networks2.G_main_vc2')

    args = parser.parse_args()
    subcmd = args.command
    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    kwargs = vars(args)
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = kwargs.pop('command')

    func_name_map = {
        'project-generated-images': 'run_projector_vc2.project_generated_images',
        'project-real-dataset-images': 'run_projector_vc2.project_real_dataset_images',
        'project-real-other-images': 'run_projector_vc2.project_real_other_images'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
