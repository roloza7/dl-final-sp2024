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

    masked_image = image_tensor * ~mask

    return masked_image, mask

class NoiseScheduler:
    def __init__(self, num_steps, beta_start = 0.0001, beta_end = 0.02, device = "cpu"):
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device, dtype=torch.float)

        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas)
        self.num_steps = num_steps
        self.text_alphas = torch.arange(num_steps, -1, -1) / num_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alphas = torch.from_numpy(self.alphas).to(self.device).float()
        self.alpha_bars = torch.from_numpy(self.alpha_bars).to(self.device).float()
        self.betas = torch.from_numpy(self.betas).to(self.device).float()
        self.text_alphas = torch.from_numpy(self.text_alphas).to(self.device).float()
        
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
