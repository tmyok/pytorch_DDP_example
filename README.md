# pytorch_DDP_example
Example of distributed dataparallel training in PyTorch.

1. Preprocess
   ```
   python3 dataset.py
   ```

2. Training
   ```
   python3 -m torch.distributed.launch --nproc_per_node=2 train.py
   ```

3. Evaluate
   ```
   python3 -m torch.distributed.launch --nproc_per_node=1 inference.py
   ```
