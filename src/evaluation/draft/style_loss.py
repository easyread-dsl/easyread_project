import argparse
import os
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vgg19


def load_image(path, device, target_size=None):
    """Load and preprocess an image for VGG."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)

    # Convert any image mode (e.g. palette, RGBA) to plain RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Handle old Pillow compatibility
    if hasattr(Image, "Resampling"):
        resample_mode = Image.Resampling.LANCZOS
    else:
        resample_mode = Image.LANCZOS

    # Resize to match target (from other image)
    if target_size is not None:
        img = img.resize(target_size, resample_mode)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor, img.size


class VGGStyleExtractor(nn.Module):
    """Extract style features from VGG19 convolutional layers."""
    def __init__(self, style_layers=None):
        super().__init__()
        if style_layers is None:
            style_layers = [0, 5, 10, 19, 28]  # conv1_1–conv5_1

        self.style_layers = style_layers

        try:
            from torchvision.models import VGG19_Weights
            weights = VGG19_Weights.IMAGENET1K_V1
            vgg = vgg19(weights=weights)
        except Exception:
            vgg = vgg19(pretrained=True)

        self.features = vgg.features

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        outputs = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.style_layers:
                outputs[i] = x
            if i >= max(self.style_layers):
                break
        return outputs


def gram_matrix(feature):
    """Compute Gram matrix for a feature tensor [B,C,H,W]."""
    b, c, h, w = feature.size()
    F = feature.view(b, c, h * w)
    G = torch.bmm(F, F.transpose(1, 2))
    return G


def style_loss_between_features(feats1, feats2, layers):
    """Compute Gatys-style loss between feature dicts."""
    total = 0.0
    for l in layers:
        f1, f2 = feats1[l], feats2[l]
        b, c, h, w = f1.shape
        G1, G2 = gram_matrix(f1), gram_matrix(f2)
        diff = G1 - G2
        El = (diff.pow(2).sum()) / (4.0 * (c ** 2) * (h * w) ** 2)
        total += El
    return total / len(layers)


def compute_style_loss(img1_path, img2_path, device=None):
    """Main function computing style loss between two images."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load first image to get reference size
    img1, size1 = load_image(img1_path, device)
    # Load second image resized to match first
    img2, _ = load_image(img2_path, device, target_size=size1)

    extractor = VGGStyleExtractor().to(device)
    extractor.eval()

    with torch.no_grad():
        feats1 = extractor(img1)
        feats2 = extractor(img2)
        loss = style_loss_between_features(feats1, feats2, extractor.style_layers)

    return loss.item()


def main():
    parser = argparse.ArgumentParser(description="Compute style loss between two images.")
    parser.add_argument("--img1", type=str, required=True, help="Path to first image")
    parser.add_argument("--img2", type=str, required=True, help="Path to second image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loss = compute_style_loss(args.img1, args.img2, device)
    print(f"\nStyle loss between '{args.img1}' and '{args.img2}': {loss:.6f}\n")


if __name__ == "__main__":
    main()
