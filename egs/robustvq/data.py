from torchvision import datasets, transforms


class CIFAR10(datasets.CIFAR10):
    def __init__(self, **kwargs):
        tr = transforms.Compose([transforms.ToTensor(), lambda x: x - 0.5])
        super(CIFAR10, self).__init__("data/cifar10", transform=tr,
                                      download=True, **kwargs)
