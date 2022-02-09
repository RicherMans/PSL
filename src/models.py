from typing import Any, Callable, List, Optional, Tuple, Union, Dict

import numpy as np
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch
from loguru import logger

import torch.nn as nn
import torchaudio
import torchaudio.transforms as audio_transforms
torchaudio.set_audio_backend('sox_io')


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0 , 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class _ConvBNReLU(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(_ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False), norm_layer(out_planes),
            nn.ReLU6(inplace=True))


class _InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        layer: torch.nn.Sequential,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(_InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
        layer = _ConvBNReLU if layer is None else layer

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(layer(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            layer(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2_DM(nn.Module):

    def __init__(self,
                 outputdim=527,
                 width_mult=1.0,
                 wavtransforms=None,
                 spectransforms=None,
                 inverted_residual_setting=None,
                 norm_layer=None,
                 **kwargs):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2_DM, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        n_mels = kwargs.get('n_mels', 64)
        n_fft = kwargs.get('n_fft', 512)
        hop_size = kwargs.get('hop_size', 160)
        win_size = kwargs.get('win_size', 512)
        f_min = kwargs.get('f_min', 0)

        input_channel = 32
        last_channel = kwargs.get('last_channel', 1280)

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(
                inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(
                                 inverted_residual_setting))

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(
            last_channel * width_mult) if width_mult > 1.0 else last_channel
        features = [
            _ConvBNReLU(1, input_channel, stride=2, norm_layer=norm_layer)
        ]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    _InvertedResidual(input_channel,
                                      output_channel,
                                      stride,
                                      expand_ratio=t,
                                      layer=_ConvBNReLU,
                                      norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(
            _ConvBNReLU(input_channel,
                        self.last_channel,
                        kernel_size=1,
                        norm_layer=norm_layer))
        features.append(nn.AdaptiveAvgPool2d((1, None)))
        # make it nn.Sequential
        self.front_end = nn.Sequential(
            audio_transforms.MelSpectrogram(f_min=f_min,
                                            sample_rate=16000,
                                            win_length=win_size,
                                            n_fft=n_fft,
                                            hop_length=hop_size,
                                            n_mels=n_mels),
            audio_transforms.AmplitudeToDB(top_db=120),
        )
        self.wavtransforms = wavtransforms if wavtransforms != None else nn.Sequential(
        )
        self.spectransforms = spectransforms if spectransforms != None else nn.Sequential(
        )

        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.last_channel, outputdim),
        )

        # weight initialization
        self.features.apply(init_weights)
        self.classifier.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Union[None, torch.Tensor]]:
        #x = B N
        if self.training:
            x = self.wavtransforms(x.unsqueeze(1)).squeeze(1)
        x = self.front_end(x)
        # B F T
        if self.training:
            x = self.spectransforms(x)
        x = rearrange(x, 'b f t -> b 1 f t')  # Add channel dim
        x = self.features(x)
        x = rearrange(x, 'b c f t -> b (f t) c')
        x = torch.sigmoid(self.classifier(x))
        return x.mean(1), x



def load_pretrained(model: torch.nn.Module, trained_model_params: dict) -> torch.nn.Module:
    model_dict = model.state_dict()
    # filter unnecessary keys
    pretrained_dict = {
        k: v
        for k, v in trained_model_params.items() if (k in model_dict) and (
            model_dict[k].shape == trained_model_params[k].shape)
    }
    assert len(pretrained_dict) > 0, "Loading failed!"
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)
    return model


if __name__ == "__main__":
    mdl = MobileNetV2_DM()
    x = torch.randn(1, 16000)
    mdl(x)
