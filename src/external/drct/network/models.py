import torch
import torch.nn as nn

import clip
import timm


def init_imagenet_weight(conv_stem_weight, input_channel=3):
    weight = conv_stem_weight
    for _ in range(1, input_channel // 3):
        weight = torch.cat([weight, conv_stem_weight], dim=1)
    return torch.nn.Parameter(weight)


class CLIPModelV2(nn.Module):
    CHANNELS = {
        "RN50": 1024,
        "ViT-B/32": 512,
        "ViT-L/14": 768,
    }

    def __init__(self, name="clip-RN50", num_classes=2, freeze_extractor=False):
        super().__init__()
        name = name.replace("clip-", "").replace("L-", "L/").replace("B-", "B/")
        self.model, self.preprocess = clip.load(name, device="cpu")
        if freeze_extractor:
            self.freeze(self.model)
        self.fc = nn.Linear(self.CHANNELS[name], num_classes)

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)
        if return_feature:
            return features
        return self.fc(features)

    @staticmethod
    def freeze(model):
        for param in model.parameters():
            param.requires_grad = False


class ContrastiveModels(nn.Module):
    def __init__(
        self,
        model_name,
        num_classes=2,
        pretrained=True,
        embedding_size=1024,
        freeze_extractor=False,
    ):
        super().__init__()
        self.model = get_models(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=embedding_size,
            freeze_extractor=freeze_extractor,
        )
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x, return_feature=False):
        feature = self.model(x)
        y_pred = self.fc(feature)
        if return_feature:
            return y_pred, feature
        return y_pred


def get_convnext(
    model_name="convnext_base_in22k",
    pretrained=True,
    num_classes=2,
    in_channel=3,
):
    net = timm.create_model(model_name, pretrained=pretrained)
    n_features = net.head.fc.in_features
    net.head.fc = nn.Linear(n_features, num_classes)

    if in_channel != 3:
        first_conv_weight = net.stem[0].weight
        first_out_channels = net.stem[0].out_channels
        first_conv = nn.Conv2d(in_channel, first_out_channels, kernel_size=4, stride=4)
        first_conv.weight = init_imagenet_weight(
            first_conv_weight, input_channel=in_channel
        )
        net.stem[0] = first_conv

    return net


def get_clip_visual_model(
    model_name="clip-ViT-L-14",
    num_classes=2,
    freeze_extractor=False,
):
    return CLIPModelV2(
        name=model_name,
        num_classes=num_classes,
        freeze_extractor=freeze_extractor,
    )


def get_models(
    model_name="convnext_base_in22k",
    pretrained=True,
    num_classes=2,
    in_channel=3,
    freeze_extractor=False,
    embedding_size=None,
):
    if embedding_size is not None and embedding_size > 0:
        return ContrastiveModels(
            model_name,
            num_classes,
            pretrained,
            embedding_size,
            freeze_extractor,
        )
    if "convnext" in model_name:
        return get_convnext(model_name, pretrained, num_classes, in_channel=in_channel)
    if "clip" in model_name:
        return get_clip_visual_model(
            model_name, num_classes, freeze_extractor=freeze_extractor
        )

    raise NotImplementedError(model_name)
