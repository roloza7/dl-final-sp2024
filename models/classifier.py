import torch
import torch.nn as nn
from models.masked_autoencoder import MaskedAEConfig, MaskedAutoEncoder
class ImageNetClassifier(nn.Module):
    def __init__(self, config: MaskedAEConfig, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes
        self.masked_ae = MaskedAutoEncoder(config)
        self.masked_ae.load_state_dict(torch.load(r"C:\Users\zacha\Downloads\\base_0"))
        # Define a classifier that maps from the encoder hidden dimension to the number of ImageNet classes
        self.classifier = nn.Sequential(
            nn.Linear(config.encoder_hidden_dim, 2048),  # Linear layer to expand the feature representation
            nn.ReLU(),  # Activation function
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(2048, 1024),  # Another linear layer
            nn.ReLU(),  # Activation function
            nn.Linear(1024, num_classes),  # Final linear layer to output the number of ImageNet classes
        )

    def forward(self, images, captions, i_position_indices=None, att_pad_mask=None):
        # Use the MaskedAutoEncoder to encode the input into latent space
        encoded = self.masked_ae(images, captions, i_position_indices, att_pad_mask)

        # We assume the relevant encoding for classification is the average of the encoded features across the sequence
        # Global Average Pooling
        pooled = torch.mean(encoded, dim=1)

        # Pass the pooled features through the classifier to get the logits for each class
        logits = self.classifier(pooled)
        return logits
