import os

from datasets.filelist_dataset import ImageFilelist


def build_dataset(dataset_name, transform, train=False):
    dataset_dir = os.getenv('DATASETS')

    # imagenet
    if dataset_name == "imagenet":
        dataset = ImageFilelist(root=dataset_dir, flist=os.path.join(dataset_dir, 'test_imagenet.txt'), transform=transform)
        return dataset

    if dataset_name == 'inaturalist':
        dataset = ImageFilelist(dataset_dir, flist=os.path.join(dataset_dir, 'test_inaturalist.txt'), transform=transform)
        return dataset

    if dataset_name == 'ninco':
        dataset = ImageFilelist(dataset_dir, flist=os.path.join(dataset_dir, 'test_ninco.txt'), transform=transform)
        return dataset

    if dataset_name == 'ssb_hard':
        dataset = ImageFilelist(dataset_dir, flist=os.path.join(dataset_dir, 'test_ssb_hard.txt'), transform=transform)
        return dataset

    if dataset_name == 'openimage_o':
        dataset = ImageFilelist(dataset_dir, flist=os.path.join(dataset_dir, 'test_openimage_o.txt'), transform=transform)
        return dataset

    if dataset_name == 'textures':
        dataset = ImageFilelist(dataset_dir, flist=os.path.join(dataset_dir, 'test_textures.txt'), transform=transform)
        return dataset

    exit(f'{dataset_name} dataset is not supported')


def get_num_classes(in_dataset_name):
    if in_dataset_name == 'cifar10':
        return 10
    if in_dataset_name == 'cifar100':
        return 100
    if in_dataset_name == 'svhn':
        return 10
    if in_dataset_name == 'imagenet':
        return 1000
    if in_dataset_name == 'imagenet30':
        return 30
    if in_dataset_name == 'imagenette':
        return 10
    exit(f'Unsupported in-dist dataset: f{in_dataset_name}')

