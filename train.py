import os
import torch

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataset import load_mnist_dataset
from distributed import setup, cleanup, reduce_tensor
from model import MnistResNet
from options import Options

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_one_epoch(model, train_loader, optimizer, criterion, rank, world_size, epoch, num_epoch):
    model.train()
    summary_loss = AverageMeter()

    if rank == 0:
        pbar = tqdm(total=len(train_loader), unit="batch")
        pbar.set_description(f"Epoch[{epoch+1}/{num_epoch}].Train")

    for imgs, labels in train_loader:
        batch_size = imgs.shape[0]

        imgs = imgs.to(rank)
        labels = labels.to(rank)
        output = model(imgs)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reduced_loss = reduce_tensor(loss, world_size)
        summary_loss.update(reduced_loss.detach().item(), batch_size)

        if rank==0:
            pbar.set_postfix({"loss":summary_loss.avg})
            pbar.update(1)

    return summary_loss

@torch.no_grad()
def validation(model, data_loader, criterion, rank, world_size, epoch, num_epoch):
    model.eval()
    summary_loss = AverageMeter()

    if rank == 0:
        pbar = tqdm(total=len(data_loader), unit="batch")
        pbar.set_description(f"Epoch[{epoch+1}/{num_epoch}].Val")

    for imgs, labels in data_loader:
        batch_size = imgs.shape[0]

        imgs = imgs.to(rank)
        labels = labels.to(rank)
        output = model(imgs)
        loss = criterion(output, labels)

        reduced_loss = reduce_tensor(loss, world_size)
        summary_loss.update(reduced_loss.detach().item(), batch_size)

        if rank==0:
            pbar.set_postfix({"loss":summary_loss.avg})
            pbar.update(1)

    return summary_loss

def training(cfg, world_size, model, dataset_train, dataset_validation):
    rank = cfg.local_rank
    batch_size = cfg.batch_size
    epochs = cfg.epoch
    lr = cfg.learning_rate
    random_seed = cfg.random_seed

    sampler = DistributedSampler(
        dataset=dataset_train,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=random_seed,
        drop_last=True)

    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        pin_memory=True,
        sampler=sampler)

    val_loader = DataLoader(
        dataset=dataset_validation,
        batch_size=batch_size,
        pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    try:
        for epoch in range(epochs):
            # In distributed mode, calling the set_epoch() method at the beginning of each epoch
            # before creating the DataLoader iterator is necessary to make shuffling work properly 
            # across multiple epochs. Otherwise, the same ordering will be always used.
            sampler.set_epoch(epoch)

            train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=opt,
                criterion=criterion,
                rank=rank,
                world_size=world_size,
                epoch=epoch,
                num_epoch=epochs)

            validation(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                rank=rank,
                world_size=world_size,
                epoch=epoch,
                num_epoch=epochs)

    except KeyboardInterrupt:
        pass

    return model

if __name__ == '__main__':
    cfg = Options().parse()
    rank = cfg.local_rank
    dataset_path = cfg.dataset_path
    output_dir = cfg.output_dir
    output_pth = cfg.output_pth

    # DDP preprocess
    world_size = setup(rank)

    # dataset
    train_dataset = load_mnist_dataset(dataset_path=dataset_path, train=True)
    val_dataset = load_mnist_dataset(dataset_path=dataset_path, train=False)

    # create model and move it to GPU with id rank
    model = MnistResNet()
    # Use torch.nn.SyncBatchNorm.convert_sync_batchnorm() to convert 
    # BatchNorm*D layer to SyncBatchNorm before wrapping Network with DDP.
    # https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    # training
    model = training(cfg, world_size, model, train_dataset, val_dataset)

    # Saving model for inference
    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/19
        torch.save(model.module.state_dict(), os.path.join(output_dir, output_pth))

    # DDP postprocess
    cleanup()