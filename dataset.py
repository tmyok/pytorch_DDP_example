import numpy as np
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from options import Options

# dataset
class ToNumpy(object):
    def __call__(self, sample):
        return np.array(sample)

def load_mnist_dataset(dataset_path, download=False, train=True):
    dataset = MNIST(root=dataset_path,
                    download=download,
                    train=train,
                    transform=torchvision.transforms.Compose(
                        [ToNumpy(), torchvision.transforms.ToTensor()])
                    )
    return dataset

if __name__ == '__main__':
    cfg = Options().parse()
    dataset_path = cfg.dataset_path

    load_mnist_dataset(dataset_path=dataset_path, download=True)