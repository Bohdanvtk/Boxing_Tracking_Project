import numpy as np


def global_ssim(x: np.ndarray, y: np.ndarray, c1: float = 0.01**2, c2: float = 0.03**2) -> float:
    """
    Global SSIM for two grayscale images (float32 in [0,1]), computed over the whole frame.

    x, y: (H, W) float32 in [0,1]

    SSIM = ((2*mu_x*mu_y + c1) * (2*cov_xy + c2)) / ((mu_x^2 + mu_y^2 + c1) * (var_x + var_y + c2))

    Returns float in [-1, 1] but in practice ~[0,1] for images.
    """
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    mu_x = float(x.mean())
    mu_y = float(y.mean())

    dx = x - mu_x
    dy = y - mu_y

    var_x = float((dx * dx).mean())
    var_y = float((dy * dy).mean())
    cov_xy = float((dx * dy).mean())

    num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)

    if den <= 1e-12:
        return 1.0  # practically identical / constant frames

    return float(num / den)
