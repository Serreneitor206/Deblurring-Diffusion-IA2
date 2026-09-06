import torch
import torch.nn as nn
import torch.optim as optim
from models import UNet, KernelGenerator
from config import Config
import os

class DeblurSDI:
    # Añadimos T y S como parámetros para poder cambiarlos en la interfaz
    def __init__(self, blur_img, kernel_size, T=None, S=None, config=None):
        self.config = config or Config
        self.blur_img = blur_img 
        self.kernel_size = kernel_size
        
        # Usar T y S dinámicos (si no se pasan, usa los del paper)
        self.T = T if T is not None else self.config.T
        self.S = S if S is not None else self.config.S
        
        self.lambda_k = self.config.LAMBDA_K
        self.mu = self.config.MU

        self.image_denoiser = UNet(in_channels=3, out_channels=3).to(blur_img.device)
        self.kernel_generator = KernelGenerator(
            kernel_size=kernel_size,
            hidden_dim=self.config.KERNEL_HIDDEN_DIM,
            n_layers=self.config.KERNEL_HIDDEN_LAYERS,
            mode='diffusion'
        ).to(blur_img.device)

        self.optimizer = optim.Adam(
            list(self.image_denoiser.parameters()) + list(self.kernel_generator.parameters()),
            lr=self.config.LR_IMAGE
        )
        self.lr_kernel = self.config.LR_KERNEL
        self.optimizer_kernel = optim.Adam(self.kernel_generator.parameters(), lr=self.lr_kernel)

        self.betas = torch.linspace(self.config.BETA_START, self.config.BETA_END, self.T).to(blur_img.device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sigmas = torch.sqrt(1 - self.alpha_bars)

        self.x_t = torch.randn_like(blur_img)
        self.z_t = torch.randn((1, self.kernel_size*self.kernel_size), device=blur_img.device)

    def load_weights(self):
        """Carga pesos previos para no empezar la optimización desde cero"""
        if os.path.exists(self.config.UNET_SAVE_PATH) and os.path.exists(self.config.KERNEL_SAVE_PATH):
            self.image_denoiser.load_state_dict(torch.load(self.config.UNET_SAVE_PATH))
            self.kernel_generator.load_state_dict(torch.load(self.config.KERNEL_SAVE_PATH))
            print("   [!] Pesos 'Warm-Start' cargados. Inferencia acelerada.")
            return True
        return False

    def save_weights(self):
        """Guarda los pesos para usarlos en el futuro"""
        torch.save(self.image_denoiser.state_dict(), self.config.UNET_SAVE_PATH)
        torch.save(self.kernel_generator.state_dict(), self.config.KERNEL_SAVE_PATH)

    def run(self):
        for t in range(self.T, 0, -1):
            sigma_t = self.sigmas[t-1].item()
            sigma_t_prime = self.mu * sigma_t

            eps_x = torch.randn_like(self.blur_img)
            eps_z = torch.randn_like(self.z_t)

            x_hat = self.x_t + sigma_t * eps_x
            z_hat = self.z_t + sigma_t_prime * eps_z

            for s in range(self.S):
                kernel = self.kernel_generator(z_hat)
                denoised = self.image_denoiser(x_hat)
                recon = self._convolve(denoised, kernel)

                loss_mse = torch.mean((recon - self.blur_img)**2)
                loss_l1 = torch.sum(torch.abs(kernel))
                loss = loss_mse + self.lambda_k * loss_l1

                self.optimizer.zero_grad()
                self.optimizer_kernel.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.optimizer_kernel.step()

            with torch.no_grad():
                self.x_t = self.image_denoiser(x_hat)
                self.z_t = self.kernel_generator(z_hat).view(1, -1)

            new_lr = max(self.lr_kernel * self.config.LR_DECAY, self.config.MIN_LR)
            for param_group in self.optimizer_kernel.param_groups:
                param_group['lr'] = new_lr
            self.lr_kernel = new_lr

        return self.x_t, self.kernel_generator(self.z_t)

    def _convolve(self, img, kernel):
        import torch.fft
        pad_h = kernel.shape[2] // 2
        pad_w = kernel.shape[3] // 2
        img_pad = torch.nn.functional.pad(img, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        H, W = img_pad.shape[-2], img_pad.shape[-1]
        K = kernel.shape[-1]
        kernel_pad = torch.zeros((1, 1, H, W), device=img.device)
        kernel_pad[:, :, pad_h:pad_h+K, pad_w:pad_w+K] = kernel
        f_img = torch.fft.fft2(img_pad, dim=(-2,-1))
        f_kernel = torch.fft.fft2(kernel_pad, dim=(-2,-1))
        f_blur = f_img * f_kernel
        blur = torch.fft.ifft2(f_blur, dim=(-2,-1)).real
        return blur[:, :, pad_h:-pad_h, pad_w:-pad_w]