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
<h4 align="center">preprint on archive: <a href="https://arxiv.org/abs/2004.04467">arXiv:2004.04467</a></h4>
<table>
  <p align="center">
  <img src="https://podgorskiy.com/static/reconstructions_multiresolution_2.jpg">  </p>
<tbody>
<tr>
<td style="padding:0;"><img src="https://user-images.githubusercontent.com/3229783/79530431-4bb90b00-803d-11ea-9ce3-25dfc3df253a.gif"></td>
<td style="padding:0;"><img src="https://user-images.githubusercontent.com/3229783/79530431-4bb90b00-803d-11ea-9ce3-25dfc3df253a.gif"></td>
</tr>
</tbody>
</table>

  <p align="center">
    <a href="#installation">Installation</a> •
    <a href="#why">Why?</a> •
    <a href="#what-is-the-performance-gain">What is the performance gain?</a> •
    <a href="#tutorial">Tutorial</a> •
    <a href="#license">License</a>
  </p>
  
<p align="center">
  <img src="https://img.shields.io/badge/pytorch-1.4.0-green.svg?style=plastic" alt="pytorch version">
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>

</p>

# ALAE


## Installation

To install requirements:

```python
pip install -r requirements.txt
```


### Installing CUDA 9.0
```
sudo echo "deb http://apt.pop-os.org/proprietary bionic main" | sudo tee -a /etc/apt/sources.list.d/pop-proprietary.list
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key 204DD8AEC33A7AFF
sudo apt update

sudo apt install system76-cuda-9.0
sudo apt install system76-cudnn-9.0
```

```
export LD_LIBRARY_PATH=/usr/lib/cuda-9.0/lib64
```

## How to Run
You need to have pytorch >= v0.4.1 and cuda/cuDNN drivers installed.

To install requirements:

```python
pip install -r requirements.txt
```

To download and prepare dataset:
```python
python prepare_celeba.py
```

To train:
```python
python VAE.py
```
