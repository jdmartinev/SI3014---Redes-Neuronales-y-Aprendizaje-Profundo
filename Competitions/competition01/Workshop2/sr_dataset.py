"""
sr_dataset.py
=============
Utilidades de dataset para super-resolución con degradación sintética.

Uso básico:
    from sr_dataset import SRDataset, show_sample_pairs, build_dataloaders

    train_ds, val_ds = SRDataset.from_dir('data/hr_images')
    train_loader, val_loader = build_dataloaders(train_ds, val_ds)
    show_sample_pairs(train_ds, n=4)
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

__all__ = ["SRDataset", "show_sample_pairs", "build_dataloaders"]

# Extensiones aceptadas
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SRDataset(Dataset):
    """
    Dataset de super-resolución con degradación sintética.

    La imagen HR se extrae como un crop aleatorio de ``hr_size × hr_size``.
    La imagen LR se genera degradando ese crop a ``lr_size × lr_size``
    con un pipeline estocástico (blur, ruido gaussiano, compresión JPEG).
    Así el par (LR, HR) siempre es geométricamente consistente.

    Args:
        paths:          Lista de rutas a las imágenes HR.
        hr_size:        Tamaño del crop de alta resolución (default 512).
        lr_size:        Tamaño de la imagen degradada (default 128).
        augment:        Si True, aplica flips horizontales y verticales al HR
                        antes de degradar.
        degrade_blur:   Probabilidad de blur gaussiano pre-downsample.
        degrade_noise:  Probabilidad de ruido gaussiano post-downsample.
        degrade_jpeg:   Probabilidad de compresión JPEG post-downsample.

    Ejemplo:
        >>> paths = sorted(Path("data/hr").rglob("*.png"))
        >>> ds = SRDataset(paths, hr_size=512, lr_size=128, augment=True)
        >>> lr_t, hr_t = ds[0]
        >>> lr_t.shape, hr_t.shape
        (torch.Size([3, 128, 128]), torch.Size([3, 512, 512]))
    """

    def __init__(
        self,
        paths: list[Path],
        hr_size: int = 512,
        lr_size: int = 128,
        augment: bool = True,
        degrade_blur: float = 0.5,
        degrade_noise: float = 0.4,
        degrade_jpeg: float = 0.5,
    ) -> None:
        super().__init__()
        if not paths:
            raise ValueError("La lista de paths está vacía.")
        self.paths = paths
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.augment = augment
        self.p_blur = degrade_blur
        self.p_noise = degrade_noise
        self.p_jpeg = degrade_jpeg
        self._to_tensor = transforms.ToTensor()

    # ------------------------------------------------------------------
    # Constructor alternativo conveniente
    # ------------------------------------------------------------------
    @classmethod
    def from_dir(
        cls,
        img_dir: str | Path,
        hr_size: int = 512,
        lr_size: int = 128,
        val_fraction: float = 0.15,
        **kwargs,
    ) -> tuple[SRDataset, SRDataset]:
        """
        Crea los splits de train y val a partir de un directorio.

        Args:
            img_dir:      Directorio raíz con las imágenes HR.
            hr_size:      Tamaño del crop HR.
            lr_size:      Tamaño de la LR degradada.
            val_fraction: Fracción del dataset para validación.
            **kwargs:     Argumentos adicionales pasados a SRDataset
                          (degrade_blur, degrade_noise, degrade_jpeg).

        Returns:
            Tupla (train_dataset, val_dataset).

        Raises:
            FileNotFoundError: Si no se encuentran imágenes en img_dir.
        """
        all_paths = sorted(
            p for p in Path(img_dir).rglob("*") if p.suffix.lower() in _IMG_EXTS
        )
        if not all_paths:
            raise FileNotFoundError(f"No se encontraron imágenes en '{img_dir}'.")

        n_val = max(1, int(len(all_paths) * val_fraction))
        train_paths = all_paths[n_val:]
        val_paths = all_paths[:n_val]

        train_ds = cls(train_paths, hr_size=hr_size, lr_size=lr_size, augment=True, **kwargs)
        val_ds = cls(val_paths, hr_size=hr_size, lr_size=lr_size, augment=False, **kwargs)

        print(f"SRDataset — train: {len(train_ds)} | val: {len(val_ds)} imágenes")
        return train_ds, val_ds

    # ------------------------------------------------------------------
    # Degradación
    # ------------------------------------------------------------------
    def _degrade(self, hr_img: Image.Image) -> Image.Image:
        """
        Aplica degradación estocástica para simular imágenes de baja resolución.

        Pipeline:
            1. Blur gaussiano (p=degrade_blur) — simula lente imperfecta.
            2. Downsample bicúbico a lr_size.
            3. Ruido gaussiano aditivo (p=degrade_noise) — simula sensor.
            4. Compresión JPEG (p=degrade_jpeg) — simula fotos de internet.
        """
        # 1. Blur pre-downsample
        if random.random() < self.p_blur:
            radius = random.uniform(0.5, 2.0)
            hr_img = hr_img.filter(ImageFilter.GaussianBlur(radius=radius))

        # 2. Downsample
        lr_img = hr_img.resize((self.lr_size, self.lr_size), Image.BICUBIC)

        # 3. Ruido gaussiano
        if random.random() < self.p_noise:
            arr = np.array(lr_img, dtype=np.float32)
            arr += np.random.randn(*arr.shape) * random.uniform(1.0, 8.0)
            lr_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        # 4. Compresión JPEG
        if random.random() < self.p_jpeg:
            quality = random.randint(60, 95)
            buf = io.BytesIO()
            lr_img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            lr_img = Image.open(buf).copy()

        return lr_img

    # ------------------------------------------------------------------
    # Interface Dataset
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        """
        Returns:
            lr_t: Tensor [3, lr_size, lr_size] en [0, 1].
            hr_t: Tensor [3, hr_size, hr_size] en [0, 1].
        """
        img = Image.open(self.paths[idx]).convert("RGB")

        # Asegurar dimensión mínima para el crop
        min_side = min(img.size)
        if min_side < self.hr_size:
            scale = self.hr_size / min_side + 0.1
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)), Image.BICUBIC
            )

        # Crop aleatorio HR
        i, j, h, w = transforms.RandomCrop.get_params(img, (self.hr_size, self.hr_size))
        hr_img = TF.crop(img, i, j, h, w)

        # Augmentación geométrica (solo flips para no distorsionar texturas)
        if self.augment:
            if random.random() > 0.5:
                hr_img = TF.hflip(hr_img)
            #if random.random() > 0.5:
            #    hr_img = TF.vflip(hr_img)

        lr_img = self._degrade(hr_img)

        return self._to_tensor(lr_img), self._to_tensor(hr_img)

    def __repr__(self) -> str:
        return (
            f"SRDataset("
            f"n={len(self)}, hr={self.hr_size}, lr={self.lr_size}, "
            f"augment={self.augment})"
        )


# ----------------------------------------------------------------------
# DataLoader factory
# ----------------------------------------------------------------------
def build_dataloaders(
    train_ds: SRDataset,
    val_ds: SRDataset,
    batch_size: int = 4,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Construye DataLoaders estándar para train y val.

    Args:
        train_ds:    Dataset de entrenamiento.
        val_ds:      Dataset de validación.
        batch_size:  Tamaño de batch (default 4).
        num_workers: Workers de prefetch (default 4).

    Returns:
        Tupla (train_loader, val_loader).
    """
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    print(
        f"DataLoaders — train: {len(train_loader)} batches | "
        f"val: {len(val_loader)} batches (batch_size={batch_size})"
    )
    return train_loader, val_loader


# ----------------------------------------------------------------------
# Visualización
# ----------------------------------------------------------------------
def show_sample_pairs(
    dataset: SRDataset,
    n: int = 4,
    title: str = "Pares LR → HR del dataset",
) -> None:
    """
    Visualiza n pares (LR, HR) del dataset para verificar la degradación.

    Args:
        dataset: Instancia de SRDataset.
        n:       Número de pares a mostrar (default 4).
        title:   Título de la figura.
    """
    n = min(n, len(dataset))
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
    if n == 1:
        axes = axes[:, None]  # mantiene indexación 2D

    for i in range(n):
        lr, hr = dataset[i]
        axes[0, i].imshow(lr.permute(1, 2, 0).clamp(0, 1))
        axes[0, i].set_title(f"LR {list(lr.shape[1:])}", fontsize=9)
        axes[0, i].axis("off")
        axes[1, i].imshow(hr.permute(1, 2, 0).clamp(0, 1))
        axes[1, i].set_title(f"HR {list(hr.shape[1:])}", fontsize=9)
        axes[1, i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
