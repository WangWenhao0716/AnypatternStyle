# Anypattern helps artists
Another application of "[AnyPattern: Towards In-context Image Copy Detection](https://arxiv.org/pdf/2404.13788.pdf)"

The text-to-image model can be used to mimic the style of artwork with little cost, and this threatens the livelihoods and creative rights of artists. To help them protect their work, we treat an artist’s ‘style’ as a ‘pattern’ and generalize the trained pattern retrieval method to identify generated images with style mimicry. 

![image](https://github.com/WangWenhao0716/AnypatternStyle/blob/main/style_demo.png)

## Training
Please refer to the original [repository](https://github.com/WangWenhao0716/AnyPattern) of AnyPattern.

## Demonstration
![image](https://github.com/WangWenhao0716/AnypatternStyle/blob/main/style_match.png)

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

## Citation
```
@inproceedings{
    wang2024AnyPattern,
    title={AnyPattern: Towards In-context Image Copy Detection},
    author={Wang, Wenhao and Sun, Yifan and Tan, Zhentao and Yang, Yi},
    booktitle={arXiv preprint arXiv:2404.13788},
    year={2024},
}
```

## Contact

If you have any questions, feel free to contact [Wenhao Wang](https://wangwenhao0716.github.io/) (wangwenhao0716@gmail.com).



