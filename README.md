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

