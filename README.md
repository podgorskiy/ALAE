<h1 align="center">
  <br>
  [CVPR2020] Adversarial Latent Autoencoders
  <br>
</h1>
  <p align="center">
    <a href="https://podgorskiy.com/">Stanislav Pidhorskyi</a> •
    <a href="https://www.statler.wvu.edu/faculty-staff/faculty/donald-a-adjeroh">Donald A. Adjeroh </a> •
    <a href="http://vision.csee.wvu.edu/~doretto/">Gianfranco Doretto</a>
  </p>
<h4 align="center">Official repository of the paper</h4>
<table>
  <p align="center">
  <img src="https://podgorskiy.com/static/reconstructions_multiresolution_2.jpg">  </p>
<tbody>
<tr>
<td style="padding:0;"><img src="https://user-images.githubusercontent.com/3229783/79670218-63080d80-818f-11ea-9e50-927b8af3e7b5.gif"></td>
<td style="padding:0;"><img src="https://user-images.githubusercontent.com/3229783/79530431-4bb90b00-803d-11ea-9ce3-25dfc3df253a.gif"></td>
</tr>
</tbody>
</table>

<p align="center">
  <img src="https://podgorskiy.com/static/stylemix.jpg">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/pytorch-1.4.0-green.svg?style=plastic" alt="pytorch version">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
</p>

  <p align="center">
    <a href="https://drive.google.com/drive/folders/1iZodDA4q1IKRRgV2nJuAyyuCwQGtL4vp?usp=sharing">Google Drive folder with models and qualitative results</a>
  </p>


# ALAE

> **Adversarial Latent Autoencoders**<br>
> Stanislav Pidhorskyi, Donald Adjeroh, Gianfranco Doretto<br>
>
> **Abstract:** *Autoencoder networks are unsupervised approaches aiming at combining generative and representational properties by learning simultaneously an encoder-generator map. Although studied extensively, the issues of whether they have the same generative power of GANs, or learn disentangled representations, have not been fully addressed. We introduce an autoencoder that tackles these issues jointly, which we call Adversarial Latent Autoencoder (ALAE). It is a general architecture that can leverage recent improvements on GAN training procedures. We designed two autoencoders: one based on a MLP encoder, and another based on a StyleGAN generator, which we call StyleALAE. We verify the disentanglement properties of both architectures. We show that StyleALAE can not only generate 1024x1024 face images with comparable quality of StyleGAN, but at the same resolution can also produce face reconstructions and manipulations based on real images. This makes ALAE the first autoencoder able to compare with, and go beyond the capabilities of a generator-only type of architecture.*

## Citation
* Stanislav Pidhorskyi, Donald A. Adjeroh, and Gianfranco Doretto. Adversarial Latent Autoencoders. In *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020. [to appear] 
>

    @InProceedings{pidhorskyi2020adversarial,
     author   = {Pidhorskyi, Stanislav and Adjeroh, Donald A and Doretto, Gianfranco},
     booktitle = {Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)},
     title    = {Adversarial Latent Autoencoders},
     year     = {2020},
     note     = {[to appear]},
    }
<h4 align="center">preprint on arXiv: <a href="https://arxiv.org/abs/2004.04467">2004.04467</a></h4>

## To run the demo

To run the demo, you will need to have a CUDA capable GPU, pytorch >= v1.3.1 and cuda/cuDNN drivers installed.
Install required packages:

    pip install -r requirements.txt
  
Download pretrained models:

    python training_artifacts/download_all.py

Run the demo:

    python interactive_demo.py

You can specify **yaml** config to use. Configs are located here: https://github.com/podgorskiy/ALAE/tree/master/configs.
By default, it uses one for FFHQ dataset.
You can change the config using `-c` parameter. To run on `celeb-hq` in 256x256 resolution, run:

    python interactive_demo.py -c celeba-hq256

However, for configs other then FFHQ, you will need to obtain new principal direction vectors for the attibutes.

## Repository organization

#### Running scripts

The code in repository is organized in such way that all scripts must be run from the root of the repository.
If you use an IDE (e.g. PyCharm or Visual Studio Code), just set *Working Directory* to point to the root of the repository.

If you want to run from command line, then you will also need to set **PYTHONPATH** variable to point to the root of the repository.

For example, let's say we've cloned repository to *~/ALAE* directory, then do:

    $ cd ~/ALAE
    $ export PYTHONPATH=$PYTHONPATH:$(pwd)

![pythonpath](https://podgorskiy.com/static/pythonpath.svg)

Now you can run scripts as follows:

    $ python style_mixing/stylemix.py

#### Repository structure


| Path | Description
| :--- | :----------
| ALAE | Repository root folder
| &boxvr;&nbsp; configs | Folder with yaml config files.
| &boxv;&nbsp; &boxvr;&nbsp; bedroom.yaml | Config file for LSUN bedroom dataset at 256x256 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; celeba.yaml | Config file for CelebA dataset at 128x128 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; celeba-hq256.yaml | Config file for CelebA-HQ dataset at 256x256 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; celeba_ablation_nostyle.yaml | Config file for CelebA 128x128 dataset for ablation study (no styles).
| &boxv;&nbsp; &boxvr;&nbsp; celeba_ablation_separate.yaml | Config file for CelebA 128x128 dataset for ablation study (separate encoder and discriminator).
| &boxv;&nbsp; &boxvr;&nbsp; ffhq.yaml | Config file for FFHQ dataset at 1024x1024 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; mnist.yaml | Config file for MNIST dataset using Style architecture.
| &boxv;&nbsp; &boxur;&nbsp; mnist_fc.yaml | Config file for MNIST dataset using only fully connected layers (Permutation Invariant MNIST).
| &boxvr;&nbsp; dataset_preparation | Folder with scripts for dataset preparation.
| &boxv;&nbsp; &boxvr;&nbsp; prepare_celeba_hq_tfrecords.py | To prepare TFRecords for CelebA-HQ dataset at 256x256 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; prepare_celeba_tfrecords.py | To prepare TFRecords for CelebA dataset at 128x128 resolution.
| &boxv;&nbsp; &boxvr;&nbsp; prepare_mnist_tfrecords.py | To prepare TFRecords for MNIST dataset.
| &boxv;&nbsp; &boxvr;&nbsp; split_tfrecords_bedroom.py | To split official TFRecords from StyleGAN paper for LSUN bedroom dataset.
| &boxv;&nbsp; &boxur;&nbsp; split_tfrecords_ffhq.py | To split official TFRecords from StyleGAN paper for FFHQ dataset.
| &boxvr;&nbsp; dataset_samples | Folder with sample inputs for different datasets. Used for figures and for test inputs during training.
| &boxvr;&nbsp; make_figures | Scripts for making various figures.
| &boxvr;&nbsp; metrics | Scripts for computing metrics.
| &boxvr;&nbsp; principal_directions | Scripts for computing principal direction vectors for various attributes. **For interactive demo**.
| &boxvr;&nbsp; style_mixing | Sample inputs and script for producing style-mixing figures.
| &boxvr;&nbsp; training_artifacts | Default place for saving checkpoints/sample outputs/plots.
| &boxv;&nbsp; &boxur;&nbsp; download_all.py | Script for downloading all pretrained models.
| &boxvr;&nbsp; interactive_demo.py | Runnable script for interactive demo.
| &boxvr;&nbsp; train_alae.py | Runnable script for training.
| &boxvr;&nbsp; train_alae_separate.py | Runnable script for training for ablation study (separate encoder and discriminator).
| &boxvr;&nbsp; checkpointer.py | Module for saving/restoring model weights, optimizer state and loss history.
| &boxvr;&nbsp; custom_adam.py | Customized adam optimizer for learning rate equalization and zero second beta.
| &boxvr;&nbsp; dataloader.py | Module with dataset classes, loaders, iterators, etc.
| &boxvr;&nbsp; defaults.py | Definition for config variables with default values.
| &boxvr;&nbsp; launcher.py | Helper for running multi-GPU, multiprocess training. Sets up config and logging.
| &boxvr;&nbsp; lod_driver.py | Helper class for managing growing/stabilizing network.
| &boxvr;&nbsp; lreq.py | Custom `Linear`, `Conv2d` and `ConvTranspose2d` modules for learning rate equalization.
| &boxvr;&nbsp; model.py | Module with high-level model definition.
| &boxvr;&nbsp; model_separate.py | Same as above, but for ablation study.
| &boxvr;&nbsp; net.py | Definition of all network blocks for multiple architectures.
| &boxvr;&nbsp; registry.py | Registry of network blocks for selecting from config file.
| &boxvr;&nbsp; scheduler.py | Custom schedulers with warm start and aggregating several optimizers.
| &boxvr;&nbsp; tracker.py | Module for plotting losses.
| &boxvr;&nbsp; utils.py | Decorator for async call, decorator for caching, registry for network blocks.


