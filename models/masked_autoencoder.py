import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedAEConfig:
    def __init__(self,
                 vocab_size,
                 hidden_dim = 768,
                 max_input_len = 256,
                 patch_size = 16,
                 output_size = (224, 224),
                 in_channels = 3,
                 n_encoder_heads = 12,
                 dim_encoder_feedforward = 3072,
                 n_encoder_layers = 6,
                 n_decoder_heads = 12,
                 dim_decoder_feedforward = 3072,
                 n_decoder_layers= 6,
                 ) -> None:
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_input_len = max_input_len
        self.patch_size = patch_size
        self.n_encoder_heads = n_encoder_heads
        self.dim_encoder_feedforward = dim_encoder_feedforward
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_heads = n_decoder_heads
        self.dim_decoder_feedforward = dim_decoder_feedforward
        self.n_decoder_layers = n_decoder_layers
        self.in_channels = in_channels
        self.prediction_sequence_length = int((output_size[0] // patch_size) * (output_size[1] // patch_size))
        self.output_size = output_size

class MaskedAEEncoder(nn.Module):
    def __init__(self, config : MaskedAEConfig) -> None:
        super().__init__()

        transformer = nn.TransformerEncoderLayer(
            config.hidden_dim,
            nhead=config.n_encoder_heads,
            dim_feedforward=config.dim_encoder_feedforward,
            activation=F.leaky_relu,
            batch_first=True
        )

        self.h = nn.TransformerEncoder(transformer, config.n_encoder_layers, norm=nn.LayerNorm(config.hidden_dim))

    def forward(self, embeddings):
        
        return self.h(embeddings)
    
class MaskedAEDecoder(nn.Module):
    def __init__(self, config : MaskedAEConfig) -> None:
        super().__init__()

        transformer = nn.TransformerEncoderLayer(
            config.hidden_dim,
            nhead=config.n_decoder_heads,
            dim_feedforward=config.dim_decoder_feedforward,
            activation=F.gelu,
            batch_first=True
        )

        self.h = nn.TransformerEncoder(transformer, config.n_decoder_layers, norm=nn.LayerNorm(config.hidden_dim))

    def forward(self, encoded):
        
        return self.h(encoded)

class MaskedAutoEncoder(nn.Module):
    def __init__(self, config : MaskedAEConfig) -> None:
        super().__init__()

        # Patch token embedding
        self.ite = nn.Linear(config.in_channels * config.patch_size ** 2, config.hidden_dim)
        # Patch positional embedding
        self.ipe = nn.Embedding(config.max_input_len, config.hidden_dim)

        self.pred_seq_len = config.prediction_sequence_length

        # Word token embedding
        self.wte = nn.Embedding(config.vocab_size, config.hidden_dim)
        # Word positional embedding
        self.wpe = nn.Embedding(config.max_input_len, config.hidden_dim)

        self.encoder = MaskedAEEncoder(config)

        self.patch_size = config.patch_size
        self.output_size = config.output_size

        self.mask_token = nn.Parameter(torch.zeros((1, 1, config.hidden_dim)))
        torch.nn.init.xavier_normal_(self.mask_token.data)

    """

    """
    def forward(self,
                images : torch.Tensor,  # (B, L, E)
                captions : torch.Tensor, # (B, L, E2)
                patch_ids : torch.Tensor = None): # <-
        
        if patch_ids == None:
            patch_ids = torch.arange(0, images.shape[1], device=images.device, dtype=torch.long)
        
        # Patch embeddings
        image_emb : torch.Tensor = self.ite(images)

        # Add positional embeddings
        image_emb = image_emb + self.ipe(patch_ids)
     
        # Token embeddings
        word_emb = self.wte(captions) # <- (B, L, hidden_dim)
        word_emb = word_emb + self.wpe(torch.arange(0, word_emb.shape[1], device=captions.device, dtype=torch.long).unsqueeze(0))

        full_emb = torch.cat([image_emb, word_emb], dim=1)
        encoded = self.encoder(full_emb)

        return encoded


class MaskedAutoEncoderForPretraining(nn.Module):
    def __init__(self, config : MaskedAEConfig) -> None:
        super().__init__()
        self.transformer = MaskedAutoEncoder(config)

        # Decoder positional embedding
        self.dpe = nn.Embedding(config.max_input_len * 2, config.hidden_dim)

        self.pred_seq_len = config.prediction_sequence_length
        self.output_size = config.output_size
        self.patch_size = config.patch_size

        self.decoder = MaskedAEDecoder(config)

        # Modelling heads
        self.lm_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, config.vocab_size)
        )

        self.im_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, config.in_channels * config.patch_size ** 2)
        )



    def forward(self, 
                images : torch.Tensor,  # (B, L, E)
                captions : torch.Tensor, # (B, L, E2)
                patch_ids : torch.Tensor, # <-
                reverse_ids : torch.Tensor):
        
        word_seq_len = captions.shape[1]
        image_seq_len = images.shape[1]

        encoded = self.transformer(images, captions, patch_ids=patch_ids)

        # Prepare sequence for decoding
        encoded_img = encoded[:, :image_seq_len, :]
        encoded_text = encoded_img[:, -word_seq_len:, :]

        mask_fill_size = (encoded_img.shape[0], self.pred_seq_len - image_seq_len, encoded_img.shape[-1])
        encoded_img = torch.cat([encoded_img, self.transformer.mask_token.expand(mask_fill_size)], dim=1)

        encoded_img = torch.gather(encoded_img, dim=1, index=reverse_ids.unsqueeze(-1).expand(encoded_img.shape))
        
        encoded_img = encoded_img + self.dpe(torch.arange(0, self.pred_seq_len, device=encoded_img.device))
        encoded_text = encoded_text + self.transformer.wpe(torch.arange(0, word_seq_len, device=encoded_text.device, dtype=torch.long))

        encoded = torch.cat([encoded_img, encoded_text], dim=1)

        decoded = self.decoder(encoded)

        image_emb = decoded[:, :self.pred_seq_len, :]
        word_emb = decoded[:, self.pred_seq_len:, :]

        captions_rec = self.lm_head(word_emb)
        # (B, L, E) -> (B, E, L) -> (B, E, H, W)
        image_rec = self.im_head(image_emb)
        
        folded_image_rec = F.fold(image_rec.permute(0, 2, 1), output_size=self.output_size, kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
        folded_image_rec = F.sigmoid(folded_image_rec)

        return folded_image_rec, captions_rec
        
"""
Example downstream model
"""
class MaskedAutoEncoderForClassification(nn.Module):
    def __init__(self, config : MaskedAEConfig):
        super().__init__()
        self.transformer = MaskedAutoEncoder(config)

    def forward(self, images, captions):

        encoded = self.transformer(images, captions) # Can optionally pass in patch ids, if not will assume [0, 1, 2, ..., 195]

        # To stuff with (B, L, E) encoded
        return None

# Captioning
# Regeneration
# Classification