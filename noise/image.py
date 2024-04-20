import torch

def mask_image(image_tensor, patch_size, masking_ratio=0.75):
    """
    Randomly masks the given percentage of image patches, returning the masked image and mask
    """
    C, H, W = image_tensor.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by the patch size."

    num_patches = (H // patch_size) * (W // patch_size)
    num_masked_patches = int(num_patches * masking_ratio)
    patch_indices = torch.randperm(num_patches)[:num_masked_patches]
    mask = torch.ones((C, H, W))

    for idx in patch_indices:
        row = (idx // (W // patch_size)) * patch_size
        col = (idx % (W // patch_size)) * patch_size
        mask[:, row:row+patch_size, col:col+patch_size] = 0
    masked_image = image_tensor * mask
    return masked_image, mask

class NoiseScheduler:
    def __init__(self, num_steps, beta_start = 0.0001, beta_end = 0.02, device = "cpu"):
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device, dtype=torch.float)

        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=-1)

    def add_noise(self, original_samples, timesteps, mask, mean=0.0, std=1.0):
        noisy_samples = None
        alpha_bars_t = self.alpha_bars[timesteps]
        if (original_samples.dim() == 3):
            alpha_bars_t = alpha_bars_t.unsqueeze(-1)
        noise = torch.randn_like(original_samples) * std + mean

        noisy_samples = torch.sqrt(alpha_bars_t).unsqueeze(-1) * original_samples + torch.sqrt(1 - alpha_bars_t).unsqueeze(-1) * noise * mask       
        return noisy_samples 