import torch
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader, Subset
from utils.data import COCOAEDataset, collate_fn
from utils.transforms import get_transform
from utils.transforms import ResizeTransform
from noise.scheduler import NoiseScheduler, LinearMaskScheduler, mask_image
from models.masked_autoencoder import MaskedAEConfig, MaskedAutoEncoderForPretraining, MaskedAutoEncoderForCaptioning, MaskedAutoEncoder
from tqdm import tqdm
import os
import sys

class ClassifierTrainer:
    def __init__(self, model, noise_scheduler, train_dataset, val_dataset, train_dataloader, val_dataloader, criterion, optimizer):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optim = optimizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
    
    def train(self, epochs):
        for epoch in (pbar := tqdm(range(2000))):
            for images, captions, lengths in train_dataloader:
                optim.zero_grad()
                images = images.to(DEVICE, non_blocking=True)
                captions = captions.to(DEVICE, non_blocking=True)
                lengths = lengths.to(DEVICE, non_blocking=True)

                masked_images, masked_text, targets, (image_positions, rp) = noise_scheduler.get_masked(images, captions, lengths, need_masks=True)
                # print(masked_images.shape, masked_text.shape)        
                reconstructed_captions = model.forward(masked_images, image_positions)
                if epoch % 25 == 0:
                    for c in captions:
                        print("Original:", train_dataset.dataset.tokenizer.decode(c))
                    for caption in reconstructed_captions:
                        # print(caption.shape)
                        values, indices = torch.topk(caption, 10)
                        # print(values, indices)
                        for i in indices:
                            print(train_dataset.dataset.tokenizer.decode(i), end = ", ")      
                        print("")                               
                # print(reconstructed_captions.shape, targets.shape)
                cap_loss = caption_loss(reconstructed_captions, targets)
                pbar.set_description(f"Epoch: {epoch}, Caption Loss : {cap_loss.item():1.3}")
                cap_loss.backward()
                # print(model.ite.weight.grad)
                optim.step()
    def train2(self, epochs):
        # caption_loss = caption_loss
        # optim = optim
        self.model.train()
        for epoch in (pbar := tqdm(range(epochs))):
            for images, captions, lengths in self.train_dataloader:
                self.optim.zero_grad()
                images = images.to(DEVICE, non_blocking=True)
                captions = captions.to(DEVICE, non_blocking=True)
                lengths = lengths.to(DEVICE, non_blocking=True)

                masked_images, text, targets, (image_positions, text_pad) = self.noise_scheduler.get_masked(images, captions, lengths, need_masks=True)
                # print(masked_images.shape, masked_text.shape)   
                reconstructed_captions = self.model.forward(masked_images, captions, text_pad, image_positions)
                if epoch % 5 == 0:
                    for c in captions:
                        print("Original:", train_dataset.dataset.tokenizer.decode(c))    
                    for c in reconstructed_captions:
                        print("Generated:", train_dataset.dataset.tokenizer.decode(torch.argmax(c, dim=-1)))                               
                shifted_original = captions[:,1:]
                shifted_reconstructed = reconstructed_captions[:,:-1]
                cap_loss = self.criterion(shifted_reconstructed.permute(0,2,1), shifted_original)
                pbar.set_description(f"Epoch: {epoch}, Caption Loss : {cap_loss.item():1.3}")
                cap_loss.backward()
                self.optim.step()
    def validate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, captions, lengths in self.val_dataloader:
                images = images.to(DEVICE, non_blocking=True)
                captions = captions.to(DEVICE, non_blocking=True)
                lengths = lengths.to(DEVICE, non_blocking=True)

                masked_images, text, targets, (image_positions, text_pad) = self.noise_scheduler.get_masked(images, captions, lengths, need_masks=True)
                # print(targets)
                # for image in images:
                #     plt.imshow(image.permute(1, 2, 0).detach().cpu().numpy())
                #     plt.show() 
                blank_captions = torch.zeros_like(captions)
                reconstructed_captions = self.model.forward(masked_images, blank_captions, text_pad, image_positions)
                for c in captions:
                    print("Original:", train_dataset.dataset.tokenizer.decode(c))    
                for c in reconstructed_captions:
                    print("Generated:", train_dataset.dataset.tokenizer.decode(torch.argmax(c, dim=-1)))                               
                
                # for caption in reconstructed_captions:
                #     values, indices = torch.topk(caption, 10)
                #     for i in indices:
                #         print(val_dataset.dataset.tokenizer.decode(i))
                cap_loss = self.criterion(reconstructed_captions.permute(0,2,1), captions)
                total_loss += cap_loss.item() * images.size(0)

        avg_loss = total_loss / len(val_dataset)
        print(f"Validation Loss: {avg_loss:.3f}")
            


if __name__ == "__main__":
    original_dataset = COCOAEDataset(root="coco/images/train2017/",
                        annFile="coco/annotations/ann2017/captions_train2017.json",
                        transform=get_transform(),
                        tokenizer=BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir='cache/'),
                        ignore_cache=False,
                        train=True)
    train_dataset = Subset(original_dataset, range(1000))
    val_dataset = Subset(original_dataset, range(1000, 1100))
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataloader = DataLoader(train_dataset,
                            batch_size=4,
                            shuffle=True,
                            collate_fn=collate_fn(train_dataset.dataset.tokenizer.pad_token_id),
                            pin_memory=True)
    val_dataloader = DataLoader(val_dataset,
                            batch_size=4,
                            shuffle=True,
                            collate_fn=collate_fn(val_dataset.dataset.tokenizer.pad_token_id),
                            pin_memory=True)

    noise_scheduler = LinearMaskScheduler(vocab_size=len(train_dataset.dataset.tokenizer), masking_ratio=0.0)

    config = MaskedAEConfig(len(train_dataset.dataset.tokenizer))
    pretrained = MaskedAutoEncoder(config).to(DEVICE)
    checkpoint = torch.load("checkpoints/base_0",map_location=DEVICE)
    pretrained.load_state_dict(checkpoint)
    model = MaskedAutoEncoderForCaptioning(MaskedAEConfig(len(train_dataset.dataset.tokenizer)), pretrained=pretrained).to(DEVICE)
    # optim = torch.optim.Adam(model.parameters(), lr=4e-5)
    optim = torch.optim.AdamW(model.parameters(), lr=1.5e-4, betas=(0.9, 0.95), weight_decay=0.03)

    image_loss = torch.nn.MSELoss()
    caption_loss = torch.nn.CrossEntropyLoss()
    # caption_loss = torch.nn.MSELoss()

    trainer = ClassifierTrainer(model = model, noise_scheduler=noise_scheduler, train_dataset=train_dataset, val_dataset=val_dataset, train_dataloader=train_dataloader, val_dataloader=val_dataloader, criterion=caption_loss, optimizer=optim)
    trainer.train2(50)
    model_path = os.path.join("checkpoints/captioning", f"model.pkl")
    torch.save(model.state_dict(), model_path)
    sys.exit()

    for epoch in (pbar := tqdm(range(2000))):
        for images, captions, lengths in train_dataloader:
            optim.zero_grad()
            images = images.to(DEVICE, non_blocking=True)
            captions = captions.to(DEVICE, non_blocking=True)
            lengths = lengths.to(DEVICE, non_blocking=True)

            masked_images, masked_text, targets, (image_positions, rp) = noise_scheduler.get_masked(images, captions, lengths, need_masks=True)
            # print(masked_images.shape, masked_text.shape)        
            reconstructed_captions = model.forward(masked_images, image_positions)
            if epoch % 25 == 0:
                for c in captions:
                    print("Original:", train_dataset.dataset.tokenizer.decode(c))
                for caption in reconstructed_captions:
                    # print(caption.shape)
                    values, indices = torch.topk(caption, 10)
                    # print(values, indices)
                    for i in indices:
                        print(train_dataset.dataset.tokenizer.decode(i), end = ", ")      
                    print("")                               
            # print(reconstructed_captions.shape, targets.shape)
            cap_loss = caption_loss(reconstructed_captions, targets)
            pbar.set_description(f"Epoch: {epoch}, Caption Loss : {cap_loss.item():1.3}")
            cap_loss.backward()
            # print(model.ite.weight.grad)
            optim.step()
            
        
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, captions, lengths in val_dataloader:
            images = images.to(DEVICE, non_blocking=True)
            captions = captions.to(DEVICE, non_blocking=True)
            lengths = lengths.to(DEVICE, non_blocking=True)

            masked_images, masked_text, targets, (image_positions, rp) = noise_scheduler.get_masked(images, captions, lengths, need_masks=True)
            print(targets)
            # for image in images:
            #     plt.imshow(image.permute(1, 2, 0).detach().cpu().numpy())
            #     plt.show() 
            reconstructed = model.forward(masked_images, lengths)
            for caption in reconstructed:
                values, indices = torch.topk(caption, 10)
                for i in indices:
                    print(val_dataset.dataset.tokenizer.decode(i))
            cap_loss = caption_loss(reconstructed, targets)
            total_loss += cap_loss.item() * images.size(0)

    avg_loss = total_loss / len(val_dataset)
    print(f"Validation Loss: {avg_loss:.3f}")

    model_path = os.path.join("checkpoints/captioning", f"model.pkl")
    torch.save(model.state_dict(), model_path)


