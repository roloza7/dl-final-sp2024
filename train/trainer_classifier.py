from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from models.classifier import ImageNetClassifier
from datasets import load_dataset
from imagenet.image_net_loader import ImageNetDataset
from models.masked_autoencoder import MaskedAEConfig
from noise.scheduler import LinearMaskScheduler
import torch
from torchvision import transforms

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Filter out pairs where either image or caption is missing
    filtered_batch = [(img, lbl) for img, lbl in zip(images, labels) if img is not None and lbl is not None]

    # Separate images and labels from the filtered batch
    images = [item[0].repeat(3, 1, 1) if item[0].shape[0] == 1 else item[0] for item in filtered_batch]
    labels = [[item[1]] for item in filtered_batch]

    images_rgb = []
    for img in images:
        if img.mode == 'CMYK':
            img = Image.fromarray(img)
            img = img.convert('RGB')
        images_rgb.append(img)
    # Transform images to tensors
    images = torch.stack(images_rgb)
    labels = torch.tensor(labels)
    # labels = torch.unsqueeze(labels, dim=1)

    return images, labels
class ClassifierTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        for name, param in self.model.named_parameters():
            # print(name, param)
            if "masked_ae" in name and not (name.endswith(".2.weight") or name.endswith(".2.bias")):
                param.requires_grad = False
    def train(self, num_epochs):
        self.model.train()
        print(f"Training... on {self.device}")
        scheduler = LinearMaskScheduler(self.model.num_classes, patch_size=self.model.masked_ae.patch_size)
        for epoch in range(num_epochs):
            total_loss = 0
            i = 0
            for data, targets in self.train_loader:
                data = scheduler.batched_linear_mask(data)[0]
                data, targets = data.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data, targets)
                targets = torch.squeeze(targets)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                i += 1
                if i % 10000 == 0:
                    print(f'Epoch {i + 1}, Loss: {total_loss / len(self.train_loader)}')
                    self.validate()
                if i % 100000 == 0:
                    torch.save(self.model.state_dict(), f'model_epoch_{i}.pth')
                    torch.save(self.optimizer.state_dict(), f'optimizer_epoch_{i}.pth')


    def validate(self):
        self.model.eval()
        total_loss = 0
        scheduler = LinearMaskScheduler(self.model.num_classes, patch_size=self.model.masked_ae.patch_size)
        with torch.no_grad():
            for data, targets in self.val_loader:
                data = scheduler.batched_linear_mask(data)[0]
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data, targets)
                targets = torch.squeeze(targets)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        print(f'Validation Loss: {total_loss / len(self.val_loader)}')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 1000  # ImageNet classes
    learning_rate = 1e-4
    batch_size = 32
    num_epochs = 1
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize as per the autoencoder's expected input
        transforms.ToTensor(),
    ])
    # Load ImageNet dataset
    ds = load_dataset("imagenet-1k")
    train_ds = ImageNetDataset(ds['train'], transform=transform)
    val_ds = ImageNetDataset(ds['validation'], transform=transform)

    # Create DataLoader instances
    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, num_workers=4)

    # Initialize the model
    model = ImageNetClassifier(MaskedAEConfig(num_classes)).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Initialize the Trainer
    trainer = ClassifierTrainer(model, train_loader, val_loader, criterion, optimizer, device)

    # Start training
    trainer.train(num_epochs=num_epochs)

if __name__ == "__main__":
    main()