# PS-SC GAN

This repository contains the main code for training a PS-SC GAN
(a GAN implemented with the Perceptual Simplicity and Spatial Constriction
constraints) introduced in the paper
[Where and What? Examining Interpretable Disentangled Representations].
The code for computing the TPL for model checkpoints from 
[disentanglemen_lib](https://github.com/google-research/disentanglement_lib)
can be found in [this](https://github.com/zhuxinqimac/TPL-Evaluate) repository.

![architecture](./imgs/traversals.gif)

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
    --metrics fid1k,tpl_small_0.3 \
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
The --C_lambda tag is the hyper-parameter for modulating the PS-loss.

## Evaluation
To evaluate a trained model, we can use the following code:
```
CUDA_VISIBLE_DEVICES=0 \
    python run_metrics.py \
    --result-dir /path/to/evaluate_results_dir \
    --network /path/to/xxx.pkl \
    --metrics fid50k,tpl_large_0.3,ppl2_wend \
    --data-dir /path/to/datasets \
    --dataset celeba_tfr \
    --include_I True \
    --mapping_nodup True \
    --num-gpus 1
```
where the --include_I is to indicate the model should be loaded with an
inference network, and --mapping_nodup is to indicate that the loaded model
has no W space duplication as in stylegan.

## Generation
We can generate random images or traversals based on a pretrained model pkl
using the following code:
```
CUDA_VISIBLE_DEVICES=0 \
    python run_generator_ps_sc.py generate-images \
    --network /path/to/xxx.pkl \
    --seeds 0-10 \
    --result-dir /path/to/gen_results_dir
```
and
```
CUDA_VISIBLE_DEVICES=0 \
    python run_generator_ps_sc.py generate-traversals \
    --network /path/to/xxx.pkl \
    --seeds 0-10 \
    --result-dir /path/to/traversal_results_dir
```

## Attributes Editing
We can conduct attributes editing with a disentangled model.
Currently we only use generated images for this experiment due to the
unsatisfactory quality of the real-image projection into
disentangled latent codes.

First we need to generate some images and put them into a directory,
e.g. /path/to/existing_generated_imgs_dir.
Second we need to assign the concepts to meaningful latent dimensions
using the --attr2idx_dict tag. For example, if the 23th dimension
represents azimuth concept, we add the item {azimuth:23} into the dictionary.
Third we need to which images to provide source attributes. We use the
--attr_source_dict tag to realize it. Note that there could be multiple
dimensions representing a single concept (e.g. in the following example
there  are 4 dimensions capturing the background information),
therefore it is more desirable to ensure the source images provide all
these dimensions (attributes) as a whole.
A source image can provide multiple attributes.
Finally we need to specify the face-source images with --face_source_ls tag.
All the face-source and attribute-source images should be located in the
--exist_imgs_dir.
An example code is as follows:
```
python run_editing_ps_sc.py \
    images-editing \
    --network /path/to/xxx.pkl \
    --result-dir /path/to/editing_results \
    --exist_imgs_dir /path/to/existing_generated_imgs_dir \
    --face_source_ls '[seed1003.png, seed1181.png, seed1191.png]' \
    --attr_source_dict '{seed1023.png: [azimuth, smile]; seed1289.png: [age,fringe]; seed1070.png: [lighting_right,lighting_left,lighting_vertical]}' \
    --attr2idx_dict '{lighting_right:33, yellowish:32, lighting_left:31, background_4:30, background_3:29, background_2: 26, background_1:18, gender:28, haircolor:27, lighting_vertical:25, clothes_color:24, azimuth:23, fringe:21, hair_style:16, smile:15, age:10, shoulder:9, glasses:8, elevation:5}'
```

## Accumulated Perceptual Distance with 2D Rotation
If a disentangled model has been trained,
the accumulated perceptual distance figures shown in Section 3.3 (and Section 8 in the Appendix)
can be plotted using the model checkpoint with the following code:
```
# Celeba
# The dimension for concepts: azimuth: 9; haircolor: 19; smile: 5; hair: 4; fringe: 11; elevation: 10; back: 18;
CUDA_VISIBLE_DEVICES=0 \
    python plot_latent_space.py \
    plot-rot-fn \
    --network /path/to/xxx.pkl \
    --seeds 1-10 \
    --latent_pair 19_5 \
    --load_gan True \
    --result-dir /path/to/acc_results/rot_19_5
```
The 2D latent traversal grid can be presented with code:
```
# Celeba
# The dimension for concepts: azimuth: 9; haircolor: 19; smile: 5; hair: 4; fringe: 11; elevation: 10; back: 18;
CUDA_VISIBLE_DEVICES=0 \
    python plot_latent_space.py \
    generate-grids \
    --network /path/to/xxx.pkl \
    --seeds 1-10 \
    --latent_pair 19_5 \
    --load_gan True \
    --result-dir /path/to/acc_results/grid_19_5
```

## Citation
```
@inproceedings{Xinqi_cvpr21,
author={Xinqi Zhu and Chang Xu and Dacheng Tao},
title={Where and What? Examining Interpretable Disentangled Representations},
booktitle={CVPR},
year={2021}
}
```
