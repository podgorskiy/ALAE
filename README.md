# Style GAN

https://arxiv.org/pdf/1812.04948.pdf

https://arxiv.org/pdf/1710.10196.pdf

Original Tensorflow code:

https://github.com/NVlabs/stylegan


Generation example (4 x Titan X for 8 hours):
<div>
	<img src='/generation.jpg'>
</div>

To install requirements:

```python
pip install -r requirements.txt
```

To download and prepare dataset:
```python
python download_mnist.py
python downscale.py
```

To train:
```python
python StyleGAN.py
```
