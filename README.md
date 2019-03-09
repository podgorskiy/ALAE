# Variational Autoencoder
Example of vanilla VAE for face image generation at resolution 128x128.

Auto-Encoding Variational Bayes: https://arxiv.org/abs/1312.6114

Generation:
<div>
	<img src='/sample_generation.jpg'>
</div>

Original Faces vs. Reconstructed Faces:

<div>
	<img src='/sample_reconstraction.jpg'>
</div>

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
