"""
train.py

We don't have to use this in our PCs but I decided it would be interesting to have a piece of code designed to work within PACEs
multi GPU environment

"""
import argparse
import torch
from torch.utils.data import random_split, Subset
import torch.nn as nn
from noise.scheduler import LinearMaskScheduler
from utils.data import COCOAEDataset, collate_fn
import signal
from time import time
from train.trainer import Trainer
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertTokenizerFast
from utils.transforms import get_transform
from models.masked_autoencoder import MaskedAutoEncoder, MaskedAutoEncoderForCaptioning, MaskedAEConfig

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script to train MMViT"
    )

    parser.add_argument("--imroot", required=True, type=str)
    parser.add_argument("--annfile", required=True, type=str)
    parser.add_argument("--save-path", required=True, type=str)
    parser.add_argument("--num-workers", required=False, type=int, default=0)
    parser.add_argument("--epochs", required=False, type=int, default=1)
    parser.add_argument("--batch-size", required=False, type=int, default=32)
    parser.add_argument("--load-path", required=False, type=str, default=None)
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    IMAGES_PATH = args.imroot
    CAPTIONS_PATH = args.annfile
    MODEL_PATH = args.save_path
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    CHECKPOINT_FILE = args.load_path
    PARALLEL = args.parallel

    if PARALLEL:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        size = int(os.environ['SLURM_NTASKS'])
        cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
        print(f"rank: {rank}, local rank: {local_rank}")
        dist.init_process_group(backend="nccl", init_method="env://", world_size=size, rank=rank)

    print("== Loading Dataset ==")
    dataset = COCOAEDataset(root="coco/images/train2017/",
                        annFile="coco/annotations/ann2017/captions_train2017.json",
                        transform=get_transform(),
                        tokenizer=BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir='cache/'),
                        ignore_cache=False,
                        train=True)
    sub_dataset = Subset(dataset, range(100))

    vocab_size = len(sub_dataset.dataset.tokenizer)
    pad_id = sub_dataset.dataset.tokenizer.pad_token_id

    train_dataset, val_dataset = random_split(sub_dataset.dataset, [0.85, 0.15])

    print("== Loading Model ==")
    config = MaskedAEConfig(vocab_size=vocab_size)
    model = MaskedAutoEncoder(config)
    distributed_state_dict = torch.load("checkpoints/base_0", map_location=torch.device('cpu'))
    keys = list(distributed_state_dict.keys())
    model.load_state_dict(distributed_state_dict)
    model = MaskedAutoEncoderForCaptioning(config, pretrained=model)
    del distributed_state_dict

    if PARALLEL:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

    image_criterion = nn.MSELoss()
    text_criterion = nn.CrossEntropyLoss()

    print("== Loading Trainer ==")
    trainer = Trainer(
        model=model,
        dataset=train_dataset,
        image_criterion=image_criterion,
        text_criterion=text_criterion,
        optimizer=torch.optim.AdamW,
        optimizer_args= {'lr': 5e-5, 'betas': (0.9, 0.95), 'weight_decay': 0.001},
        lr_sched=torch.optim.lr_scheduler.CosineAnnealingLR,
        lr_sched_args= {'eta_min': 0},
        noise_scheduler=LinearMaskScheduler(vocab_size, masking_ratio=0),
        validation_dataset=val_dataset,
        collate_fn=collate_fn(pad_id),
        parallel=PARALLEL,
        num_replicas=size if PARALLEL else None,
        rank=rank if PARALLEL else None,
        local_rank=local_rank if PARALLEL else None,
    )

    # def usr1_sig_handler(signum, frame):
    #     trainer.interrupt = True

    # Signal handler for when pace decides we're done, comment this on Windows (no SIGUSR1)
    # signal.signal(signal.SIGUSR1, usr1_sig_handler)

    print("== Training Start ==")
    trainer.train(N_EPOCHS, 
                  BATCH_SIZE,
                  save_path=MODEL_PATH,
                  num_workers=NUM_WORKERS,
                  val_batch_size=BATCH_SIZE,
                  load_path=CHECKPOINT_FILE) 
    print("== Training End ==")
