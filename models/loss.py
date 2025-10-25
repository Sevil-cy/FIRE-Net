import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch import autograd as autograd


"""
Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace)
      (2*): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace)
      (7*): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace)
      (16*): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (17): ReLU(inplace)
      (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace)
      (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (24): ReLU(inplace)
      (25*): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
      (26): ReLU(inplace)
      (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): ReLU(inplace)
      (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (31): ReLU(inplace)
      (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (33): ReLU(inplace)
      (34*): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (35): ReLU(inplace)
      (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
"""


# --------------------------------------------
# Perceptual loss
# --------------------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=[2,7,16,25,34], use_input_norm=True, use_range_norm=False):
        super(VGGFeatureExtractor, self).__init__()
        '''
        use_input_norm: If True, x: [0, 1] --> (x - mean) / std
        use_range_norm: If True, x: [0, 1] --> x: [-1, 1]
        '''
        model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        self.use_range_norm = use_range_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.list_outputs = isinstance(feature_layer, list)
        if self.list_outputs:
            self.features = nn.Sequential()
            feature_layer = [-1] + feature_layer
            for i in range(len(feature_layer)-1):
                self.features.add_module('child'+str(i), nn.Sequential(*list(model.features.children())[(feature_layer[i]+1):(feature_layer[i+1]+1)]))
        else:
            self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])

        print(self.features)

        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_range_norm:
            x = (x + 1.0) / 2.0
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        if self.list_outputs:
            output = []
            for child_model in self.features.children():
                x = child_model(x)
                output.append(x.clone())
            return output
        else:
            return self.features(x)


class PerceptualLoss(nn.Module):
    """VGG Perceptual loss
    """

    def __init__(self, feature_layer=[2,7,16,25,34], weights=[0.1,0.1,1.0,1.0,1.0], lossfn_type='l1', use_input_norm=True, use_range_norm=False):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(feature_layer=feature_layer, use_input_norm=use_input_norm, use_range_norm=use_range_norm)
        self.lossfn_type = lossfn_type
        self.weights = weights
        if self.lossfn_type == 'l1':
            self.lossfn = nn.L1Loss()
        else:
            self.lossfn = nn.MSELoss()
        print(f'feature_layer: {feature_layer}  with weights: {weights}')

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x_vgg, gt_vgg = self.vgg(x), self.vgg(gt.detach())
        loss = 0.0
        if isinstance(x_vgg, list):
            n = len(x_vgg)
            for i in range(n):
                loss += self.weights[i] * self.lossfn(x_vgg[i], gt_vgg[i])
        else:
            loss += self.lossfn(x_vgg, gt_vgg.detach())
        return loss

# --------------------------------------------
# GAN loss: gan, ragan
# --------------------------------------------
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        elif self.gan_type == 'softplusgan':
            def softplusgan_loss(input, target):
                # target is boolean
                return F.softplus(-input).mean() if target else F.softplus(input).mean()

            self.loss = softplusgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type in ['wgan', 'softplusgan']:
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


# --------------------------------------------
# TV loss
# --------------------------------------------
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        """
        Total variation loss
        https://github.com/jxgu1016/Total_Variation_Loss.pytorch
        Args:
            tv_loss_weight (int):
        """
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# --------------------------------------------
# Charbonnier loss
# --------------------------------------------
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss



def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.
        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (
        path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.
    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.
    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty


# --------------------------------------------
# Thermal Fidelity Loss
# --------------------------------------------

class InfernoMapper(nn.Module):
    """
    Convert RGB inferno-mapped thermal images into a 1-channel temperature proxy map in [0,1].
    This mapper is fixed (non-trainable).
    """
    def __init__(self, device="cuda"):
        super().__init__()
        # Create inferno LUT (256 samples)
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("inferno")
        lut = torch.tensor(cmap(torch.linspace(0, 1, 256))[:, :3], dtype=torch.float32, device=device)
        self.register_buffer("lut_rgb", lut)  # [256,3]
        # Precompute RGB→index linear mapping by least squares
        K = 256
        X = torch.cat([lut, torch.ones(K, 1, device=device)], dim=1)  # [256,4]
        y = torch.linspace(0, 1, K, device=device).unsqueeze(1)       # [256,1]
        W = torch.linalg.lstsq(X, y).solution.squeeze(1)              # [4]
        self.register_buffer("w_rgb", W[:3])
        self.b = float(W[3])

    @torch.no_grad()
    def forward(self, img_rgb):
        """
        img_rgb: [B,3,H,W] in [0,1] — Inferno-mapped RGB thermal image
        return: [B,1,H,W] temperature proxy map in [0,1]
        """
        B, C, H, W = img_rgb.shape
        x = img_rgb.view(B, 3, -1).permute(0, 2, 1)  # [B,HW,3]
        # Linear projection to index
        k = (x @ self.w_rgb) + self.b  # [B,HW]
        k = k.clamp(0.0, 1.0)
        tmap = k.view(B, 1, H, W)
        # Optional local normalization per-image (stabilize dynamic range)
        tmin = tmap.amin(dim=[2, 3], keepdim=True)
        tmax = tmap.amax(dim=[2, 3], keepdim=True)
        tmap = (tmap - tmin) / (tmax - tmin + 1e-6)
        return tmap


class ThermalFidelityLoss(nn.Module):
    """
    Three-part fidelity-aware loss for thermal images:
        LT    : temperature-proxy MAE
        Lstat : low-order statistic consistency (mean/std)
        Lmono : monotonic / rank consistency
    """

    def __init__(self, lambda_T=0.1, lambda_stat=0.05, lambda_rank=0.02,
                 num_pairs=1024, device="cuda"):
        super().__init__()
        self.mapper = InfernoMapper(device=device)
        self.lam_T = lambda_T
        self.lam_stat = lambda_stat
        self.lam_rank = lambda_rank
        self.num_pairs = num_pairs

    def temperature_proxy(self, img_rgb):
        return self.mapper(img_rgb)

    def LT(self, T_pred, T_gt):
        """Temperature-proxy MAE"""
        return F.l1_loss(T_pred, T_gt)

    def Lstat(self, T_pred, T_gt, win=16):
        """Low-order statistics (mean/std) over local window"""
        pad = win // 2
        mean_p = F.avg_pool2d(T_pred, win, stride=1, padding=pad)
        mean_g = F.avg_pool2d(T_gt, win, stride=1, padding=pad)
        std_p = torch.sqrt(F.avg_pool2d(T_pred ** 2, win, stride=1, padding=pad) - mean_p ** 2 + 1e-6)
        std_g = torch.sqrt(F.avg_pool2d(T_gt ** 2, win, stride=1, padding=pad) - mean_g ** 2 + 1e-6)
        Lm = (mean_p - mean_g).abs().mean()
        Ls = (std_p - std_g).abs().mean()
        return Lm + Ls

    def Lmono(self, T_pred, T_gt):
        """Monotonic / rank consistency"""
        B, _, H, W = T_pred.shape
        N = min(self.num_pairs, H * W)
        # flatten
        Tp = T_pred.view(B, -1)
        Tg = T_gt.view(B, -1)
        # sample pairs
        i = torch.randint(0, H * W, (B, N), device=Tp.device)
        j = torch.randint(0, H * W, (B, N), device=Tp.device)
        # compute sign consistency
        sign = (Tp.gather(1, i) - Tp.gather(1, j)) * (Tg.gather(1, i) - Tg.gather(1, j))
        Lmono = F.relu(-sign).mean()
        return Lmono

    def forward(self, pred_rgb, gt_rgb):
        """
        pred_rgb, gt_rgb: [B,3,H,W], RGB inferno-mapped thermal images
        return: total_loss, dict of each term
        """
        # temperature proxy conversion
        T_pred = self.temperature_proxy(pred_rgb)
        with torch.no_grad():
            T_gt = self.temperature_proxy(gt_rgb)

        # each loss
        LT = self.LT(T_pred, T_gt)
        Lstat = self.Lstat(T_pred, T_gt)
        Lmono = self.Lmono(T_pred, T_gt)

        total = (self.lam_T * LT +
                 self.lam_stat * Lstat +
                 self.lam_rank * Lmono)

        return total, {"LT": LT.item(), "Lstat": Lstat.item(), "Lmono": Lmono.item()}


# --------------------------------------------
# Combined L1 + Thermal Loss (Direct Addition)
# --------------------------------------------

class CombinedL1ThermalLoss(nn.Module):
    """
    Combined L1 loss and Thermal Fidelity loss with direct addition
    Uses the internal weights already defined in ThermalFidelityLoss
    """
    
    def __init__(self, l1_weight=1.0, thermal_kwargs=None):
        super().__init__()
        self.l1_weight = l1_weight
        thermal_kwargs = thermal_kwargs or {}
        
        self.l1_loss = nn.L1Loss()
        self.thermal_loss = ThermalFidelityLoss(**thermal_kwargs)
        
    def forward(self, pred, gt):
        """
        pred, gt: [B,3,H,W] RGB images
        return: total_loss, loss_dict
        """
        l1_loss = self.l1_loss(pred, gt)
        thermal_loss, thermal_details = self.thermal_loss(pred, gt)
        
        total_loss = self.l1_weight * l1_loss + thermal_loss
        
        loss_dict = {
            "l1_loss": l1_loss.item(),
            "thermal_loss": thermal_loss.item(),
            "total_loss": total_loss.item(),
            **thermal_details
        }
        
        return total_loss, loss_dict


class CombinedL1ThermalSimpleLoss(nn.Module):
    """
    Simplified version that only returns the total loss (no dict)
    Direct addition without extra thermal_weight
    """
    
    def __init__(self, l1_weight=1.0, **thermal_kwargs):
        super().__init__()
        self.l1_weight = l1_weight
        self.l1_loss = nn.L1Loss()
        self.thermal_loss = ThermalFidelityLoss(**thermal_kwargs)
        
    def forward(self, pred, gt):
        l1_loss = self.l1_loss(pred, gt)
        thermal_loss, _ = self.thermal_loss(pred, gt)
        return self.l1_weight * l1_loss + thermal_loss