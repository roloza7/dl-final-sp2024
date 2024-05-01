from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from tqdm import trange
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
    filtered_batch = [(img, lbl) for img, lbl in zip(images, labels) if img is not None and lbl is not None and img.shape[0] == 3]

    # Separate images and labels from the filtered batch
    images = [item[0] for item in filtered_batch]
    labels = [[item[1]] for item in filtered_batch]

    images = torch.stack(images)
    labels = torch.tensor(labels)

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
            total_correct = 0
            total_samples = 0
            i = 0
            with trange(len(self.train_loader), unit="batch") as pbar:
                for data, targets in self.train_loader:
                    data = scheduler.batched_linear_mask(data)[0]
                    data, targets = data.to(self.device), targets.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model.forward(data) # Learnable captions
                    targets = torch.squeeze(targets)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == targets).sum().item()
                    total_samples += targets.size(0)
                    accuracy = total_correct / total_samples
                    i += 1
                    pbar.update(1)
                    pbar.set_description(f'Loss: {total_loss / i}, Accuracy: {accuracy * 100:.2f}%')
                self.validate()
                torch.save(self.model.state_dict(), f'model_epoch_{epoch}.pth')
                torch.save(self.optimizer.state_dict(), f'optimizer_epoch_{epoch}.pth')


    def validate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        scheduler = LinearMaskScheduler(self.model.num_classes, patch_size=self.model.masked_ae.patch_size)

        with torch.no_grad(), trange(len(self.val_loader), unit="batch") as pbar:
            for data, targets in self.val_loader:
                data = scheduler.batched_linear_mask(data)[0]
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model.forward(data)

                # Calculate loss
                targets = torch.squeeze(targets)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)

                accuracy = total_correct / total_samples

                pbar.update(1)
                pbar.set_description(f'Validation Loss: {total_loss / len(self.val_loader)}, Validation Accuracy: {accuracy * 100:.2f}%')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-4
    batch_size = 32
    num_epochs = 25
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
    model = ImageNetClassifier(MaskedAEConfig(30522)).to(device)
    model.load_state_dict(torch.load(r"C:\Users\zacha\Documents\GitHub\dl-final-sp2024\model_epoch_0.pth"))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(torch.load(r"C:\Users\zacha\Documents\GitHub\dl-final-sp2024\optimizer_epoch_0.pth"))

    # Initialize the Trainer
    trainer = ClassifierTrainer(model, train_loader, val_loader, criterion, optimizer, device)

    # Start training
    trainer.train(num_epochs=num_epochs)

if __name__ == "__main__":
    main()