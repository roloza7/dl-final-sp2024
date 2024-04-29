import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from models.classifier import ImageNetClassifier
from datasets import load_dataset
from torchvision import transforms
from imagenet.image_net_loader import ImageNetDataset
from models.masked_autoencoder import MaskedAEConfig
class ClassifierTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        for name, param in self.model.named_parameters():
            if "masked_ae" in name and not (name.endswith(".2.weight") or name.endswith(".2.bias")):
                param.requires_grad = False
    def train(self, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for data, targets in self.train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(data, targets)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {total_loss / len(self.train_loader)}')
            torch.save(self.model.state_dict(), f'model_epoch_{epoch + 1}.pth')
            self.validate()

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

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