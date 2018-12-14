# A PyTorch implementation of Deformable ConvNets v2
This repository contains code for **Deformable ConvNets v2 (Modulated Deformable Convolution)** based on [Deformable ConvNets v2: More Deformable, Better Results
](https://arxiv.org/abs/1811.11168) implemented in PyTorch. This implementation of deformable convolution based on [ChunhuanLin/deform_conv_pytorch](https://github.com/ChunhuanLin/deform_conv_pytorch), thanks to ChunhuanLin.

## Requirements
- Python 3.6
- PyTorch 0.4

## Usage
Replace regular convolution (following model's conv2) with modulated deformable convolution:
```python
class ConvNet(nn.Module):
  def __init__(self):
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d((2, 2))
    self.avg_pool = nn.AdaptiveAvgPool2d(1)

    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.conv2 = nn.DeformConv2d(32, 64, 3, padding=1, modulation=True)

    self.fc = nn.Linear(64, 10)

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.pool(x)
    x = self.relu(self.conv2(x))

    x = self.avg_pool(x)
    x = x.view(x.shape[0], -1)
    x = self.fc(x)

    return x
```

## Training
### ScaledMNIST
ScaledMNIST is randomly scaled MNIST.

Use modulated deformable convolution at conv3~4:
```
python train.py --arch ScaledMNISTNet --deform True --modulation True --min-deform-layer 3
```
Use deformable convolution at conv3~4:
```
python train.py --arch ScaledMNISTNet --deform True --modulation False --min-deform-layer 3
```
Use only regular convolution:
```
python train.py --arch ScaledMNISTNet --deform False --modulation False
```

## TODO
 - [ ] Results of ScaledMNIST experiments
 - [x] Initialize weight of modulated deformable convolution based on paper
 - [x] Learning rates of offset and modulation are set to different values from other layers.
 - [ ] DeepLab + DCNv2

<!--
## Results
| Model                                           |   Error rate (%)  | Loss |
|:------------------------------------------------|:-----------------:|:----:|
| WideResNet28-10 baseline                        |              4.70 | 0.193|
| WideResNet28-10 +RICAP                          |              3.94 | 0.162|
| WideResNet28-10 +Random Erasing                 |              4.36 | 0.163|
| WideResNet28-10 +Mixup                          |              4.09 | 0.232|
| WideResNet28-10 baseline (original paper)       |              3.89 |     -|
| WideResNet28-10 +RICAP (original paper)         |              2.85 |     -|
| WideResNet28-10 +Random Erasing (original paper)|              4.65 |     -|
| WideResNet28-10 +Mixup (original paper)         |              3.02 |     -|

Learning curves of loss and accuracy.

![loss](loss.png)

![acc](acc.png)
-->
