from train.trainer import Trainer
import torch.nn as nn


# trainer = Trainer(
#     module,               <- module we want to train
#     dataset=dataset,      <- dataset to train on
#     criterion,            <- loss function
#     optimizer,            <- optimizer CLASS i.e. torch.optim.AdamW
#     optimizer_args,       <- dict containing optimizer args i.e. {'lr':1e-3} 
#     validation_dataset,   <- dataset to validate on (optional)
#     lr_sched,             <- lr scheduler (optional, not implemented)
#     lr_sched_args,        <- dict containing lr sched args (optional, not implemented)
#     collate_fn,           <- collate_fn (optional, we might want to use if our dataset needs custom batching (it probably does))
#     parallel,             <- whether to use multiple GPUs (defaults to false)
#     num_replicas,         <- number of GPUs (only needed if parallel)
#     rank,                 <- rank of this GPU (ask me (Rodrigo) if you want to set up distributed) (only needed if parallel)
# )

# trainer.train(self,
#               n_epochs        <- TOTAL number of epochs to train
#               batch_size      <- training batch size
#               save_path       <- path to save our checkpoints (required, it's always going to save it somewhere to avoid lost progress)
#               num_workers     <- number of cpu cores (in distributed, number of cpu cores PER GPU) (optional)
#               val_batch_size  <- validation batch size (optional, defaults to batch_size)
#               start_epoch     <- what epoch we are starting on (only used if loading from a previous checkpoint) (optional)
#               load_path,      <- file to load previous checkpoint from (optional)
#)

# TODO: I'll write some code to make this work further, 
# this is first commit is just boilerplate that I'll edit later once we have a good dataset/image setup