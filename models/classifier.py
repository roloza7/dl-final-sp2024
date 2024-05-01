import torch
import torch.nn as nn
from models.masked_autoencoder import MaskedAEConfig, MaskedAutoEncoder


class ImageNetClassifier(nn.Module):
    def __init__(self, config: MaskedAEConfig, num_classes=1000, caption_length=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.num_classes = num_classes
        self.caption_length = caption_length
        self.masked_ae = MaskedAutoEncoder(config)
        self.masked_ae.load_state_dict(torch.load(r"C:\Users\zacha\Downloads\\base_0"))

        # Define a classifier that maps from the encoder hidden dimension to the number of ImageNet classes
        self.classifier = nn.Sequential(
            nn.Linear(config.encoder_hidden_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

        # Initialize a trainable caption parameter
        self.trainable_captions = nn.Parameter(torch.randn(1, self.caption_length))

    def forward(self, images, i_position_indices=None, att_pad_mask=None):
        # if use_trainable_captions:
        #     # Expand the trainable captions to match the batch size of images
        #     captions = self.trainable_captions.expand(images.size(0), -1).to(self.device)
        # else:
            # Placeholder for actual captions, this should be replaced with real data if not using trainable captions
        captions = torch.zeros(images.size(0), self.caption_length).to(self.device)

        # Encode the input into latent space using the MaskedAutoEncoder
        encoded = self.masked_ae(images, captions, i_position_indices, att_pad_mask)

        # Global Average Pooling
        pooled = torch.mean(encoded, dim=1)

        # Get the logits for each class
        logits = self.classifier(pooled)
        return logits
