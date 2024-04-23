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
from noise.scheduler import LinearMaskScheduler
import time
import os

class Trainer():
    def __init__(self,
                 model : nn.Module,
                 dataset : torch.utils.data.Dataset,
                 image_criterion : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 text_criterion : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 optimizer : torch.optim.Optimizer,
                 optimizer_args : dict,
                 noise_scheduler : LinearMaskScheduler,
                 validation_dataset : torch.utils.data.Dataset = None,
                 lr_sched : torch.optim.lr_scheduler._LRScheduler = None,
                 lr_sched_args : dict = None,
                 collate_fn = None,
                 parallel : bool = False,
                 num_replicas : int = None,
                 rank : int = None,
                 local_rank : int = None) -> None:
        
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
        self.image_criterion = image_criterion
        self.text_criterion = text_criterion
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
        self.local_rank = local_rank
        self.head = rank == 0 or self.parallel == False

        self.logger = TrainLogger("dl-train", "logs/") if self.head else None

        self.noise_scheduler = noise_scheduler

        self.val_loss = []
        self.train_loss = []


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
              load_path : str = None):
        
        scaler = GradScaler()

        state_dict = {
            'start_time': time.time()
        }
        
        # Load running stats
        if load_path != None:
            stats = torch.load("running_stats.pkl")
            if self.head:
                self.val_loss = stats['val_loss']
                self.train_loss = stats['train_loss']
            self.epoch = stats['epoch']
        else:
            self.epoch = 0

        # Enable distributed training
        if self.parallel:
            if self.local_rank == None:
                raise ValueError("Rank must be defined for distributed training")
            self.model = self.model.to(self.local_rank)
            self.model = DDP(self.model, device_ids=[self.local_rank])
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
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

        # Get datasets
        train_dataloader, val_dataloader = self.__prepare_dataloaders(batch_size, num_workers, val_batch_size)

        # Optional lr scheduler

        if self.lr_sched != None:
            lr_scheduler : torch.optim.lr_scheduler._LRScheduler = self.lr_sched(optimizer, T_max=len(train_dataloader) , last_epoch=-1, **self.lr_sched_args)


        if self.head:
            self.logger.log("Starting training...")

        for epoch in range(self.epoch, n_epochs):
            self.model.train()
            for hook in self.epoch_start_hooks:
                hook(state_dict)
            if self.head:
                self.logger.log_start_epoch(epoch)

            epoch_image_loss = torch.tensor([0], device=self.device, dtype=torch.float)
            epoch_caption_loss = torch.tensor([0], device=self.device, dtype=torch.float)
            epoch_image_loss.requires_grad = False
            epoch_caption_loss.requires_grad = False

            for images, captions in train_dataloader:

                if self.interrupt == True:
                    break

                
                # TODO: Loading X variables here
                images = images.to(self.device, non_blocking=True)
                captions = captions.to(self.device, non_blocking=True)

                masked_images, masked_text, (ip, rp, _) = self.noise_scheduler.get_masked(images, captions, need_masks=True)

                optimizer.zero_grad(set_to_none=True)
                # Automatic reduced precision, makes transformers faster
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    # Model forward step here
                    reconstructed_images, reconstructed_captions = self.model.forward(masked_images, masked_text, ip, rp)
                    img_loss = self.image_criterion(reconstructed_images, images)
                    txt_loss = self.text_criterion(reconstructed_captions.permute(0, 2, 1), captions)
                    loss = img_loss + txt_loss
                    
                epoch_image_loss += img_loss.detach() / len(train_dataloader)
                epoch_caption_loss += txt_loss.detach() / len(train_dataloader)

                # This is needed due to reduced precision, don't worry about it (or ask me)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                if self.lr_sched != None:
                    lr_scheduler.step()
                scaler.update()

            # Gathering loss data (this is just for analytics)
            if self.parallel:
                dist.all_reduce(epoch_image_loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(epoch_caption_loss, op=dist.ReduceOp.AVG)

            if self.head:
                self.logger.log(f"Finished epoch {epoch}, image loss {epoch_image_loss.item():1.5}, caption loss {epoch_caption_loss.item():1.5}")
                self.train_loss.append(epoch_image_loss.item())
                self.val_loss.append(epoch_caption_loss.item())

            # Optional validation
            if self.validate:
                val_img_loss, val_txt_loss = self.eval(val_dataloader)
                if self.head:
                    self.logger.log(f"Finished epoch {epoch}, validation image loss {val_img_loss.item():1.5}, validation caption loss {val_txt_loss.item():1.5}")
                    self.val_loss.append(val_img_loss.item())
                    self.val_loss.append(val_txt_loss.item())
                if self.lr_sched != None:
                    lr_scheduler.step(val_txt_loss + val_img_loss)

            if self.head:
                self.logger.log(f"Saving Checkpoint...")
                torch.save(self.model.state_dict(), f"{save_path}_{epoch}.chkp")
                torch.save({'val_loss': self.val_loss, 'train_loss': self.train_loss, 'epoch': epoch }, f"running_stats.pkl")

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

        epoch_image_loss = torch.tensor([0], device=self.device, dtype=torch.float)
        epoch_caption_loss = torch.tensor([0], device=self.device, dtype=torch.float)
        epoch_image_loss.requires_grad = False
        epoch_caption_loss.requires_grad = False

        with torch.no_grad():
            for images, captions in val_dataloader:
                
                images = images.to(self.device, non_blocking=True)
                captions = captions.to(self.device, non_blocking=True)

                masked_images, masked_text, (ip, rp, _) = self.noise_scheduler.get_masked(images, captions, need_masks=True)

                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    reconstructed_images, reconstructed_captions = self.model.forward(masked_images, masked_text, ip, rp)
                    img_loss = self.image_criterion(reconstructed_images, images)
                    txt_loss = self.text_criterion(reconstructed_captions.permute(0, 2, 1), captions)
                    
                epoch_image_loss += img_loss.cpu().item() / len(val_dataloader)
                epoch_caption_loss += txt_loss.cpu().item() / len(val_dataloader)                    

        # Gather loss data
        if self.parallel:
            dist.all_reduce(epoch_image_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(epoch_caption_loss, op=dist.ReduceOp.AVG)

        return epoch_image_loss, epoch_caption_loss