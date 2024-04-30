import torch
import torch.nn.functional as F

def mask_image(image_tensor, patch_size, masking_ratio=0.75):
    """
    Randomly masks the given percentage of image patches, returning the masked image and mask
    """

    if len(image_tensor.shape) < 4:
        image_tensor = image_tensor.unsqueeze(0)

    B, C, H, W = image_tensor.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by the patch size."

    mask_H, mask_W = int(H // patch_size), int(W // patch_size)

    mask = torch.rand((B, mask_H, mask_W), device=image_tensor.device) < masking_ratio

    mask = mask.repeat_interleave(patch_size, dim=1, output_size=H).repeat_interleave(patch_size, dim=2, output_size=W)

    if len(mask.shape) < 4:
        mask = mask.unsqueeze(1)

    masked_image = image_tensor * ~mask

    return masked_image, mask

class LinearMaskScheduler:
    def __init__(self, vocab_size, masking_ratio = 0.75, patch_size = 16, pad_token_id = 0, text_skip = 2):
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        self.masking_ratio = masking_ratio
        self.pad_token_id = pad_token_id
        self.text_skip = text_skip

    def batched_linear_mask(self, image_tensor : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(image_tensor.shape) < 4:
            image_tensor = image_tensor.unsqueeze(0)
        
        B, _, H, W = image_tensor.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "Image dimensions must be divisible by the patch size."

        folded = F.unfold(image_tensor, (self.patch_size, self.patch_size), padding=0, stride=(self.patch_size, self.patch_size)).permute(0, 2, 1)

        idx_to_keep = int(folded.shape[1] * (1 - self.masking_ratio))

        # torch doesn't have randperm for batches for some godforsaken reason
        permutations = torch.rand((folded.shape[:2]), device=image_tensor.device).argsort(dim=-1)
        shuffle_backward = permutations.argsort(dim=-1)

        shuffle_forward = permutations[:, :idx_to_keep]
        masked = torch.gather(folded, dim=1, index=shuffle_forward.unsqueeze(-1).expand((shuffle_forward.shape) + (folded.shape[-1],)))

        return folded, shuffle_backward
    
    def batched_text_linear_mask(self, captions, lengths : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = captions.shape[0]

        max_length = torch.max(lengths)
        pad_mask = torch.arange(0, max_length, device=captions.device)[None, :] >= lengths
        
        permutations = torch.rand((captions.shape[:]), device=captions.device)
        permutations[pad_mask] = torch.iinfo(torch.long).max
        permutations = permutations.argsort(dim=-1)
        # permutations = permutations[:, ::2]
        
        corruptions = torch.gather(captions, dim=1, index=permutations)

        # Change permutations to permutations [:, 1::2] for no-reconstruction loss
        text_targets = torch.zeros((B, self.vocab_size), device=captions.device, dtype=torch.long).scatter_(dim=1, index=captions, src=torch.ones_like(captions)).float()
        # Ignore padding tokens
        text_targets[:, 0] = 0
        
        return corruptions, (~pad_mask).contiguous().float(), text_targets
    
    def get_masked(self, image_tensor, captions, lengths, need_masks = False):
        B = image_tensor.shape[0]
        assert image_tensor.shape[0] == captions.shape[0], "Image batch size must equal caption batch size"
    
        masked_images, shuffle_backward = self.batched_linear_mask(image_tensor)
        masked_text, text_pad_mask, text_targets = self.batched_text_linear_mask(captions, lengths)

        if need_masks:
            return masked_images, masked_text, text_targets, (shuffle_backward, text_pad_mask)

        return masked_images, masked_text, text_targets
    
class NoiseScheduler:
    def __init__(self, num_steps, beta_start = 0.0001, beta_end = 0.02, device = "cpu"):
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device, dtype=torch.float)

        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, 0)
        self.num_steps = num_steps
        self.text_alphas = torch.arange(num_steps, -1, -1) / num_steps

        self.device = device
        self.alphas = self.alphas.to(self.device).float()
        self.alpha_bars = self.alpha_bars.to(self.device).float()
        self.betas = self.betas.to(self.device).float()
        self.text_alphas = self.text_alphas.to(self.device).float()
        
    def add_noise(self, original_samples, timesteps, mask, mean=0.0, std=1.0):
        noisy_samples = None
        alpha_bars_t = self.alpha_bars[timesteps]
        if (original_samples.dim() == 3):
            alpha_bars_t = alpha_bars_t.unsqueeze(-1)
        noise = torch.randn_like(original_samples) * std + mean

        noisy_samples = torch.sqrt(alpha_bars_t).unsqueeze(-1) * original_samples + torch.sqrt(1 - alpha_bars_t).unsqueeze(-1) * noise * mask       
        return noisy_samples 
    
    def add_text_noise(self, captions, timesteps, vocab_size):
        '''Apply random masking to input based on timestep
        
        INPUTS:
        captions: tensor tokens of captions shape(batch_size, N)
        timesteps: tensor shape (batch_size)
        vocab_size: int
        
        RETURNS:
        corruptions: tensor shape (batch_size, N)'''
        alpha = 1 - self.text_alphas[timesteps]
        alpha = torch.unsqueeze(alpha, dim=1).repeat(1, captions.shape[1])
        m = torch.bernoulli(alpha).int()
        noise = torch.randint(1, vocab_size+1, size=captions.shape)

        corruptions = (1 - m) * captions + m * noise

        return corruptions
