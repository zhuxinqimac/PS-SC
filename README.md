# PS-SC GAN

This repository contains the main code for training a PS-SC GAN
(a GAN implemented with the Perceptual Simplicity and Spatial Constriction
constraints) introduced in the paper
[Where and What? Examining Interpretable Disentangled Representations].

## Requirements

* Python == 3.7.2
* Numpy == 1.19.1
* TensorFlow == 1.15.0
* This code is based on [StyleGAN2](https://github.com/NVlabs/stylegan2) which
relies on custom TensorFlow ops that are compiled on the fly using
[NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).
To test that your NVCC installation is working correctly, run:

```.bash
nvcc test_nvcc.cu -o test_nvcc -run
| CPU says hello.
| GPU says hello.
```

## Preparing datasets

**CelebA**.
To prepare the tfrecord version of CelebA dataset, first download the original aligned-and-cropped version
from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, then use the following code to
create tfrecord dataset:
```
python dataset_tool.py create_celeba /path/to/new_tfr_dir /path/to/downloaded_celeba_dir
```
For example, the new_tfr_dir can be: datasets/celeba_tfr.

**FFHQ**.
We use the 512x512 version which can be directly downloaded from
the Google Drive [link](https://drive.google.com/u/0/uc?export=download&confirm=aAOF&id=1M-ulhD5h-J7sqSy5Y1njUY_80LPcrv3V)
using browser. Or the file can be downloaded using the official script from
[Flickr-Faces-HQ](https://github.com/NVlabs/ffhq-dataset).
Put the xxx.tfrecords file into a two-level directory such as: datasets/ffhq_tfr/xxx.tfrecords.

**Other Datasets**.
The tfrecords versions of DSprites and 3DShapes datasets can be produced
```
python dataset_tool.py create_subset_from_dsprites_npz /path/to/new_tfr_dir /path/to/dsprites_npz
```
and
```
python dataset_tool.py create_subset_from_shape3d /path/to/new_tfr_dir /path/to/shape3d_file
```
See dataset_tool.py for how other datasets can be produced.

## Training
To train a model on CelebA with 2 GPUs, run code:
```
CUDA_VISIBLE_DEVICES=0,1 \
    python run_training_ps_sc.py \
    --result-dir /path/to/results_ps_sc/celeba \
    --data-dir /path/to/datasets \
    --dataset celeba_tfr \
    --metrics fid1k,tpl_small \
    --num-gpus 2 \
    --mirror-augment True \
    --model_type ps_sc_gan \
    --C_lambda 0.01 \
    --fmap_decay 1 \
    --epsilon_loss 3 \
    --random_seed 1000 \
    --random_eps True \
    --latent_type normal \
    --batch_size 8 \
    --batch_per_gpu 4 \
    --n_samples_per 7 \
    --return_atts True \
    --I_fmap_base 10 \
    --G_fmap_base 9 \
    --G_nf_scale 6 \
    --D_fmap_base 10 \
    --fmap_min 64 \
    --fmap_max 512 \
    --topk_dims_to_show -1 \
    --module_list '[Const-512, ResConv-up-1, C_spgroup-4-5, ResConv-id-1, Noise-2, ResConv-up-1, C_spgroup-4-5, ResConv-id-1, Noise-2, ResConv-up-1, C_spgroup-4-5, ResConv-id-1, Noise-2, ResConv-up-1, C_spgroup-4-5, ResConv-id-1, Noise-2, ResConv-up-1, C_spgroup-4-5, ResConv-id-1, Noise-2, ResConv-id-2]'
```
Note that for the dataset directory we need to separate
the path into --data-dir and --dataset tags.
The --model_type tag only specifies the PS-loss, and 
we need to use the C_spgroup-n_squares-n_codes in the --module_list tag 
to specify where to insert the Spatial Constriction modules in the generator.
The latent traversals and metrics will be logged in the resulting directory.
