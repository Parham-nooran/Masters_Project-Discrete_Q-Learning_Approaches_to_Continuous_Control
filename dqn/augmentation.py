import torch
import torch.nn.functional as F


def random_shift(images, pad_size=4):
    """Apply random shift augmentation to images."""
    n, c, h, w = images.shape

    # Pad images
    padded = F.pad(images, [pad_size] * 4, mode="replicate")

    # Random crop back to original size
    top = torch.randint(0, 2 * pad_size + 1, (n,))
    left = torch.randint(0, 2 * pad_size + 1, (n,))

    shifted_images = torch.zeros_like(images)
    for i in range(n):
        shifted_images[i] = padded[i, :, top[i] : top[i] + h, left[i] : left[i] + w]

    return shifted_images
