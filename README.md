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
<h4 align="center">Official implementation of the paper</h4>
<h4 align="center">A native extension for Python built with C++ and <a href="https://github.com/pybind/pybind11" target="_blank">pybind11</a>.</h4>

  <p align="center">
    <a href="#installation">Installation</a> •
    <a href="#why">Why?</a> •
    <a href="#what-is-the-performance-gain">What is the performance gain?</a> •
    <a href="#tutorial">Tutorial</a> •
    <a href="#license">License</a>
  </p>
  
<p align="center">
  <a href="https://badge.fury.io/py/dareblopy"><img src="https://badge.fury.io/py/dareblopy.svg" alt="PyPI version" height="18"></a>
  <a href="https://pepy.tech/project/dareblopy"><img src="https://pepy.tech/badge/dareblopy"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="https://api.travis-ci.com/podgorskiy/bimpy.svg?branch=master"><img src="https://travis-ci.org/podgorskiy/DareBlopy.svg?branch=master"></a>
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
