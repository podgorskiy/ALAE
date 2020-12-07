Scripts in this folder are for fining principal directions for attributes in W space.
These directions are used by script `interactive_demo.py`

Committed files `direction_*.npy` are computed for the model ffhq/model_157.pth 
To run `interactive_demo.py` with other models, you will need to find new directions:

```shell script
python printipal_directions/generate_images.py
python printipal_directions/extract_attributes.py
python printipal_directions/find_printipal_directions.py
```
