import torch
import torch.nn as nn
import torch.nn.functional as F

# class MaskedAEConfig:
#     def __init__(self,
#                  vocab_size,
#                  encoder_hidden_dim = 1024,
#                  decoder_hidden_dim = 512,
#                  max_input_len = 256,
#                  patch_size = 16,
#                  output_size = (224, 224),
#                  in_channels = 3,
#                  n_encoder_heads = 16,
#                  dim_encoder_feedforward = 3072,
#                  n_encoder_layers = 18,
#                  n_decoder_heads = 8,
#                  dim_decoder_feedforward = 3072,
#                  n_decoder_layers = 2,
#                  ) -> None:
        

class MaskedAEConfig:
    def __init__(self,
                 vocab_size,
                 encoder_hidden_dim = 1024,
                 decoder_hidden_dim = 1024,
                 max_input_len = 256,
                 patch_size = 16,
                 output_size = (224, 224),
                 in_channels = 3,
                 n_encoder_heads = 16,
                 dim_encoder_feedforward = 3072,
                 n_encoder_layers = 16,
                 n_decoder_heads = 8,
                 dim_decoder_feedforward = 3072,
                 n_decoder_layers = 10,
                 max_caption_length = 10,
                 ) -> None:
        
        self.vocab_size = vocab_size
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.max_input_len = max_input_len
        self.patch_size = patch_size
        self.n_encoder_heads = n_encoder_heads
        self.dim_encoder_feedforward = dim_encoder_feedforward
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_heads = n_decoder_heads
        self.dim_decoder_feedforward = dim_decoder_feedforward
        self.n_decoder_layers = n_decoder_layers
        self.max_caption_length = max_caption_length
        self.in_channels = in_channels
        self.prediction_sequence_length = int((output_size[0] // patch_size) * (output_size[1] // patch_size))
        self.output_size = output_size

class MaskedAEEncoder(nn.Module):
    def __init__(self, config : MaskedAEConfig) -> None:
        super().__init__()

        transformer = nn.TransformerEncoderLayer(
            config.encoder_hidden_dim,
            nhead=config.n_encoder_heads,
            dim_feedforward=config.dim_encoder_feedforward,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )

        self.h = nn.TransformerEncoder(transformer, config.n_encoder_layers, norm=nn.LayerNorm(config.encoder_hidden_dim), enable_nested_tensor=False)

    def forward(self, embeddings, att_pad_mask = None):
        
        return self.h(embeddings, src_key_padding_mask=att_pad_mask)
    
class MaskedAEDecoder(nn.Module):
    def __init__(self, config : MaskedAEConfig) -> None:
        super().__init__()

        transformer = nn.TransformerEncoderLayer(
            config.decoder_hidden_dim,
            nhead=config.n_decoder_heads,
            dim_feedforward=config.dim_decoder_feedforward,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )

        self.h = nn.TransformerEncoder(transformer, config.n_decoder_layers, norm=nn.LayerNorm(config.decoder_hidden_dim), enable_nested_tensor=False)

    def forward(self, encoded):
        
        return self.h(encoded)

class MaskedAutoEncoder(nn.Module):
    def __init__(self, config : MaskedAEConfig) -> None:
        super().__init__()

        # Patch token embedding
        self.ite = nn.Linear(config.in_channels * config.patch_size ** 2, config.encoder_hidden_dim)
        # Patch positional embedding
        self.ipe = nn.Embedding(config.max_input_len, config.encoder_hidden_dim)

        # Word token embedding
        self.wte = nn.Embedding(config.vocab_size, config.encoder_hidden_dim)

        self.encoder = MaskedAEEncoder(config)

        self.patch_size = config.patch_size
        self.output_size = config.output_size

        self.mask_token = nn.Parameter(torch.zeros((1, 1, config.encoder_hidden_dim)))
        torch.nn.init.xavier_normal_(self.mask_token.data)

    """

    """
    def forward(self,
                images : torch.Tensor,  # (B, L, E)
                captions : torch.Tensor = None, # (B, L, E2)
                i_position_indices : torch.Tensor = None,
                att_pad_mask : torch.Tensor = None): # <-
        
        if i_position_indices == None:
            i_position_indices = torch.arange(0, images.shape[1], device=images.device, dtype=torch.long)
        
        # Patch embeddings
        image_emb : torch.Tensor = self.ite(images)

        # Add positional embeddings
        image_emb = image_emb + self.ipe(i_position_indices)
     
        # Token embeddings (no positional embedding anymore since we're doing tags in v3)
        
        if captions is not None:
            
            word_emb = self.wte(captions) # <- (B, L, hidden_dim)
            full_emb = torch.cat([image_emb, word_emb], dim=1)
        else:
            full_emb = image_emb
        encoded = self.encoder(full_emb, att_pad_mask)

        return encoded


class MaskedAutoEncoderForPretraining(nn.Module):
    def __init__(self, config : MaskedAEConfig) -> None:
        super().__init__()
        self.transformer = MaskedAutoEncoder(config)

        # Decoder positional embedding
        self.decoder_patch_pos = nn.Embedding(config.max_input_len, config.encoder_hidden_dim)
        self.word_cls_token = nn.Parameter(torch.zeros(1, 1, config.encoder_hidden_dim))

        self.pred_seq_len = config.prediction_sequence_length
        self.output_size = config.output_size
        self.patch_size = config.patch_size

        self.transform = nn.Linear(config.encoder_hidden_dim, config.decoder_hidden_dim)

        self.decoder = MaskedAEDecoder(config)

        # Modelling heads
        self.lm_head = nn.Sequential(
            nn.Linear(config.decoder_hidden_dim, 3072),
            nn.LeakyReLU(0.2),
            nn.Linear(3072, config.vocab_size)
        )

        self.im_head = nn.Sequential(
            nn.Linear(config.decoder_hidden_dim, 3072),
            nn.LeakyReLU(0.2),
            nn.Linear(3072, config.in_channels * config.patch_size ** 2)
        )



    def forward(self, 
                images : torch.Tensor,  # (B, L, E)
                captions : torch.Tensor, # (B, L, E2)
                text_mask : torch.Tensor, # <-
                reverse_ids : torch.Tensor):
        
        B = captions.shape[0]

        image_seq_len = images.shape[1]

        image_attentions = torch.ones((B, image_seq_len), device=images.device, dtype=torch.float)
        total_attentions = torch.cat([image_attentions, text_mask], dim=1)
        total_attentions = (1 - total_attentions) * torch.finfo(torch.float).min

        encoded = self.transformer(images, captions, i_position_indices=reverse_ids[:, :image_seq_len], att_pad_mask = total_attentions)

        # Prepare sequence for decoding
        encoded_img = encoded[:, :image_seq_len, :]
        encoded_text = encoded[:, image_seq_len:, :]

        mask_fill_size = (encoded_img.shape[0], self.pred_seq_len - image_seq_len, encoded_img.shape[-1])
        encoded_img = torch.cat([encoded_img, self.transformer.mask_token.expand(mask_fill_size)], dim=1)

        encoded_img = torch.gather(encoded_img, dim=1, index=reverse_ids.unsqueeze(-1).expand(encoded_img.shape))
        
        encoded_img = encoded_img + self.decoder_patch_pos(torch.arange(0, self.pred_seq_len, device=encoded_img.device))

        encoded = torch.cat([encoded_img, encoded_text, self.word_cls_token.expand(B, 1, encoded_text.shape[-1])], dim=1)

        encoded = self.transform(encoded)

        decoded = self.decoder(encoded)

        image_emb = decoded[:, :self.pred_seq_len, :]
        # (B, decoder_hidden_dim)
        word_cls = decoded[:, -1, :]

        # (B, vocab_size)
        captions_rec = self.lm_head(word_cls)
        captions_rec = F.sigmoid(captions_rec)

        # (B, L, E) -> (B, E, L)
        image_rec = self.im_head(image_emb)

        # (B, L, E) -> (B, C, H, W)
        folded_image_rec = F.fold(image_rec.permute(0, 2, 1), output_size=self.output_size, kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
        folded_image_rec = F.sigmoid(folded_image_rec)

        return folded_image_rec, captions_rec
        
"""
Example downstream model
"""
class MaskedAutoEncoderForGeneration(nn.Module):
    def __init__(self, config : MaskedAEConfig):
        super().__init__()
        self.transformer = MaskedAutoEncoder(config)

        self.img_cls = nn.Parameter(torch.empty(1, 1, config.in_channels * config.patch_size ** 2))
        self.img_dcd = nn.Parameter(torch.empty(1, config.prediction_sequence_length, config.encoder_hidden_dim))
        torch.nn.init.xavier_normal_(self.img_dcd.data)
        torch.nn.init.xavier_normal_(self.img_cls.data)

        self.image_token_len = config.prediction_sequence_length
        self.output_size = config.output_size
        self.patch_size = config.patch_size

        decoder_layer = nn.TransformerDecoderLayer(
            config.encoder_hidden_dim,
            config.n_encoder_heads,
            dim_feedforward=config.dim_encoder_feedforward,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            config.n_decoder_layers,
            norm=nn.LayerNorm(config.encoder_hidden_dim)
        )
        self.im_head = nn.Sequential(
            nn.Linear(config.encoder_hidden_dim, 3072),
            nn.LeakyReLU(0.2),
            nn.Linear(3072, config.in_channels * config.patch_size ** 2),
            nn.Sigmoid()
        )

    # Removing images, reverse ids just for backwards compatibility
    def forward(self, captions, text_mask):

        cls_size = (captions.shape[0], 1, self.img_cls.shape[-1])

        cls_text_mask = torch.cat([torch.ones(captions.shape[0], 1, device=captions.device), text_mask], dim=1)
        cls_text_mask = (1 - cls_text_mask) * torch.finfo(torch.float).min

        # (B, L, E)
        memory = self.transformer(self.img_cls.expand(cls_size), captions, att_pad_mask = cls_text_mask)

        img_dcd_size = (captions.shape[0],) + self.img_dcd.shape[1:]

        decoded = self.decoder(self.img_dcd.expand(img_dcd_size), memory, memory_key_padding_mask = cls_text_mask)

        image = self.im_head(decoded)

        image = F.fold(image.permute(0, 2, 1), output_size=self.output_size, kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))

        return image


# Captioning
    
class MaskedAutoEncoderForCaptioning(nn.Module):
    def __init__(self, config: MaskedAEConfig, pretrained=None):
        super().__init__()
        if pretrained == None:
            self.transformer = MaskedAutoEncoder(config)
        else:
            self.transformer = pretrained

        self.dummy_input = nn.Parameter(torch.empty(1, 1, config.encoder_hidden_dim))
        torch.nn.init.xavier_normal_(self.dummy_input.data)

        self.wte = nn.Embedding(config.vocab_size, config.encoder_hidden_dim)

        self.text_emb = nn.Parameter(torch.empty(1, config.max_caption_length, config.encoder_hidden_dim))
        torch.nn.init.xavier_normal_(self.text_emb.data)

        decoder_layer = nn.TransformerDecoderLayer(
            config.decoder_hidden_dim,
            config.n_decoder_heads,
            dim_feedforward=config.dim_decoder_feedforward,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            config.n_decoder_layers,
            norm=nn.LayerNorm(config.decoder_hidden_dim)
        )
        
        self.caption_head = nn.Sequential(
            nn.Linear(config.encoder_hidden_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, config.vocab_size),
            nn.Softmax(dim=1)
        )

    def forward(self, images, captions, pad_mask, image_positions):
        encoded_images = self.transformer(images)
        text_mask = pad_mask
        causal_mask = torch.tril(torch.ones(text_mask.shape[1], text_mask.shape[1], device=captions.device), diagonal=0)
        caption_embed = self.wte(captions)
        decoded_captions = self.decoder(tgt=caption_embed, memory=encoded_images, tgt_mask = causal_mask, tgt_key_padding_mask=text_mask, tgt_is_causal=True)
        captions = self.caption_head(decoded_captions)
        return captions    