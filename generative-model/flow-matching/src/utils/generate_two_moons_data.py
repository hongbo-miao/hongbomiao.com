import torch


def generate_two_moons_data(sample_count: int, noise: float = 0.05) -> torch.Tensor:
    """
    Generate two interleaving half-circles (two moons) dataset.

    This is a classic 2D dataset for testing generative models because:
    - It has a non-trivial, multi-modal structure
    - It is easy to visualize
    - Simple distributions (like single Gaussian) cannot represent it

    Args:
        sample_count: Total number of samples to generate
        noise: Standard deviation of Gaussian noise added to points

    Returns:
        Tensor of shape (sample_count, 2) containing 2D points

    """
    samples_per_moon = sample_count // 2

    theta_upper = torch.linspace(0, torch.pi, samples_per_moon)
    upper_moon_x = torch.cos(theta_upper)
    upper_moon_y = torch.sin(theta_upper)
    upper_moon = torch.stack([upper_moon_x, upper_moon_y], dim=1)

    theta_lower = torch.linspace(0, torch.pi, sample_count - samples_per_moon)
    lower_moon_x = 1 - torch.cos(theta_lower)
    lower_moon_y = -torch.sin(theta_lower) + 0.5
    lower_moon = torch.stack([lower_moon_x, lower_moon_y], dim=1)

    data = torch.cat([upper_moon, lower_moon], dim=0)

    data = data + torch.randn_like(data) * noise

    shuffle_indices = torch.randperm(sample_count)
    return data[shuffle_indices]
