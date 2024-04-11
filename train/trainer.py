import torch.utils
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.nn as nn
import torch
import torch.utils.data
from typing import Callable
from train.log import TrainLogger
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os

class Trainer():
    def __init__(self,
                 model : nn.Module,
                 dataset : torch.utils.data.Dataset,
                 criterion : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 optimizer : torch.optim.Optimizer,
                 optimizer_args : dict,
                 validation_dataset : torch.utils.data.Dataset = None,
                 lr_sched : torch.optim.lr_scheduler._LRScheduler = None,
                 lr_sched_args : dict = None,
                 collate_fn = None,
                 parallel : bool = False,
                 num_replicas : int = None,
                 rank : int = None) -> None:
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model
        self.model = model

        # Dataset
        self._dataset = dataset
        self._validation_dataset = validation_dataset
        # Run validation dataset after each epoch
        self.validate = validation_dataset != None
        # Collate function (might be useful if we use tokenization)
        self.collate_fn = collate_fn
        # Arguments for sampler (if any)
        if parallel is True:
            if num_replicas is None: 
                raise ValueError("num_replicas must be defined if distributed training is enabled")
            if rank is None:
                raise ValueError("rank must be defined if distributed training is enabled")
            self.sampler_args = {'num_replicas': num_replicas, 'rank': rank, 'shuffle': True}

        # Loss and optimizer
        self.criterion = criterion
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        self.epoch_end_hooks = []
        self.epoch_start_hooks = []
        self.interrupt = False

        # Parallel tools
        self.parallel = parallel
        self.rank = rank
        self.head = rank == 0 or self.parallel == False

        self.logger = TrainLogger("dl-rain", "logs/") if self.head else None


    """
    Prepare dataloders for training
    """
    def __prepare_dataloaders(self, batch_size : int, num_workers : int = 0, val_batch_size : int = None) -> tuple[DataLoader, DataLoader]:
        
        train_dataloader = DataLoader(
            self._dataset,
            batch_size=batch_size,
            sampler=DistributedSampler(self._dataset, **self.sampler_args) if self.parallel is True else None,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )

        val_dataloader = None
        if self.validate is True:
            if val_batch_size == None:
                val_batch_size = batch_size
            
            val_dataloader = DataLoader(
                self._validation_dataset,
                batch_size=val_batch_size,
                sampler=DistributedSampler(self._validation_dataset, **self.sampler_args) if self.parallel is True else None,
                num_workers=num_workers,
                collate_fn=self.collate_fn,
                pin_memory=True
            )
        return train_dataloader, val_dataloader
    
    def train(self,
              n_epochs : int,
              batch_size : int,
              save_path : str,
              num_workers : int = 0,
              val_batch_size : int = None,
              start_epoch : int = 0,
              load_path : str = None):
        
        scaler = GradScaler()

        state_dict = {
            'start_time': time.time()
        }
        
        # Enable distributed training
        if self.parallel:
            if self.rank == None:
                raise ValueError("Rank must be defined for distributed training")
            self.model = self.model.to(self.rank)
            self.model = DDP(self.model, device_ids=[self.rank])
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            if load_path != None:
                self.model.load_state_dict(
                    torch.load(load_path, map_location=map_location)
                )
            # Let all models synchronize
            dist.barrier()
        # Single GPU training
        else:
            if load_path != None:
                self.model.load_state_dict(
                    torch.load(load_path)
                )

        # Create optimizer with params we just created
        optimizer : torch.optim.Optimizer = self.optimizer(self.model.parameters(), **self.optimizer_args)
        # Optional lr scheduler
        lr_scheduler : torch.optim.lr_scheduler._LRScheduler = self.lr_sched(optimizer, **self.lr_sched_args)

        # Get datasets
        train_dataloader, val_dataloader = self.__prepare_dataloaders(batch_size, num_workers, val_batch_size)

        if self.head:
            self.logger.log("Starting training...")

        for epoch in range(start_epoch, n_epochs):
            self.model.train()
            for hook in self.epoch_start_hooks:
                hook(state_dict)

            self.logger.log_start_epoch(epoch)

            epoch_loss = torch.Tensor([0])
            epoch_loss.requires_grad = False

            for x_batch in train_dataloader:

                if self.interrupt == True:
                    break

                
                # TODO: Loading X variables here
                

                optimizer.zero_grad(set_to_none=True)
                # Automatic reduced precision, makes transformers faster
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    # Model forward step here
                    loss = None
                    
                epoch_loss += loss.cpu().item() / len(train_dataloader)

                # This is needed due to reduced precision, don't worry about it (or ask me)
                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()

            # Gathering loss data (this is just for analytics)
            if self.parallel:
                dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)

            if self.head:
                self.logger.log(f"Finished epoch {epoch}, train loss {epoch_loss.item():1.5}")

            # Optional validation
            if self.validate:
                val_loss = self.eval(val_dataloader)
                if self.head:
                    self.logger.log(f"Finished epoch {epoch}, validation loss {val_loss.item():1.5}")

            if self.head:
                self.logger.log(f"Saving Checkpoint...")
                torch.save(self.model.state_dict(), f"{save_path}_{epoch}.chkp")

            if self.parallel:
                # Sync everyone again
                dist.barrier()
                self.model.load_state_dict(
                    torch.load(f"{save_path}_{epoch}.chkp", map_location=map_location)
                )

            for hook in self.epoch_end_hooks:
                hook(state_dict)

    def eval(self, val_dataloader : DataLoader):
        self.model.eval()

        total_loss = torch.Tensor([0])
        total_loss.requires_grad = False

        # Getting number of items in dataset, we'll divide by this to get mean loss
        N = len(self._validation_dataset)

        with torch.no_grad():
            for x_batch in val_dataloader:
                
                # TODO: Load X variables here

                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    # Inference step here
                    loss = None

                    total_loss += loss.cpu().item()

        # Gather loss data
        if self.parallel:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)

        return total_loss.item()