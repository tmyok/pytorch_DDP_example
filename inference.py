import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import load_mnist_dataset
from distributed import setup, cleanup
from model import MnistResNet
from options import Options

@torch.no_grad()
def evaluate(model, data_loader, rank):
    model.eval()

    cnt = 0
    total = len(data_loader.dataset)

    if rank == 0:
        data_loader = tqdm(data_loader)

    for imgs, labels in data_loader:
        imgs = imgs.to(rank)
        labels = labels.to(rank)

        output = model(imgs)
        predict = torch.argmax(output, dim=1)
        cnt += (predict == labels).sum().item()

    return cnt / total

if __name__ == '__main__':
    cfg = Options().parse()
    batch_size = cfg.batch_size
    dataset_path = cfg.dataset_path
    output_dir = cfg.output_dir
    output_pth = cfg.output_pth
    rank = cfg.local_rank

    # DDP preprocess
    world_size = setup(rank)

    # dataset
    dataset = load_mnist_dataset(
        dataset_path=dataset_path,
        train=False)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True)

    # create model and move it to GPU with id rank
    model = MnistResNet()
    # Use torch.nn.SyncBatchNorm.convert_sync_batchnorm() to convert 
    # BatchNorm*D layer to SyncBatchNorm before wrapping Network with DDP.
    # https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.load_state_dict(torch.load(os.path.join(output_dir, output_pth)))
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    # evaluate DDP model
    accuracy = evaluate(model, data_loader, rank)

    if rank == 0:
        print(f'accuracy: {accuracy}')

    # DDP postprocess
    cleanup()