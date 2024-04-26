# AnypatternStyle
The style extractor trained on AnyPattern


## Installation
```
conda create -n style python=3.9
conda activate style
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install timm==0.4.12
```


## Usage

```python
import requests
import torch
from PIL import Image

from anypattern_style_extractor import preprocessor, create_model

model_name = 'vit_base_pattern'
weight_name = 'vit_ddpmm_8gpu_512_torch2_ap31_pattern.pth.tar'
model = create_model(model_name, weight_name)

url = "https://huggingface.co/datasets/WenhaoWang/AnyPattern/resolve/main/Irises.jpg"
image = Image.open(requests.get(url, stream=True).raw)
x = preprocessor(image).unsqueeze(0)

style_features = model.forward_features(x)  # => torch.Size([1, 768])
style_features_normalized = torch.nn.functional.normalize(style_features, p=2, dim=1)  # => torch.Size([1, 768])

```
