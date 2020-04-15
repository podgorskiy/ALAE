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
