import torchvision.transforms as transforms
import torch
from models.densenet import DenseNet3
from models.mobilenetv2 import MobileNetV2
from models.resnet_cifar import ResNet34
from models.resnet_imagenet import ResNet50, ResNet101
from models.vgg16 import vgg16
from models.convnext import convnext_base
from models.densenet_imagenet import densenet121
from models.vit import vit_b_16, vit_b_32, vit_l_16, vit_l_32
from models.wideresnet import WideResNet
from models.my_dino_v2 import dinov2_vits14_lc, dinov2_vitb14_lc, dinov2_vitl14_lc, dinov2_vitg14_lc
from models.swin_transformer import swin_b, swin_s
from models.mlp_mixer import mlp_mixer_b_16, mlp_mixer_l_16


def build_model(model_name, num_classes):
    if model_name == 'resnet34':
        model = ResNet34(num_classes)
        transform = transforms.Compose([
            transforms.CenterCrop(size=(32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return model, transform
    if model_name == 'resnet50':
        model = ResNet50()
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform
    if model_name == 'resnet101':
        model = ResNet101()
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform
    if model_name == 'mobilenetv2':
        model = MobileNetV2(num_classes)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform
    if model_name == 'densenet100':
        model = DenseNet3(100, int(num_classes))
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            # transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
        ])
        return model, transform
    if model_name == 'wideresnet40':
        model = WideResNet(depth=40, num_classes=int(num_classes), widen_factor=2)
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            # transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
        ])
        return model, transform
    if model_name == 'vgg16':
        model = vgg16(pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform
    if model_name == 'convnext':
        model = convnext_base(pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform
    if model_name == 'densenet_imagenet':
        model = densenet121(pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform
    if model_name == 'vit-b':
        model = vit_b_16(pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform
    if model_name == 'vit-l':
        model = vit_l_16(pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform
    if model_name == 'swin-s':
        model = swin_s(pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform

    if model_name == 'swin-b':
        model = swin_b(pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform
    if model_name == 'dinov2':
        model = dinov2_vitg14_lc(pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform
    if model_name == 'mlp_mixer_b_16':
        model = mlp_mixer_b_16()
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform
    if model_name == 'mlp_mixer_l_16':
        model = mlp_mixer_l_16()
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return model, transform
    exit('{} model is not supported'.format(model_name))
