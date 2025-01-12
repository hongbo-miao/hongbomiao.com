# https://docs.sunpy.org/en/stable/generated/gallery/map/image_bright_regions_gallery_example.html

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import sunpy.map
from scipy import ndimage
from sunpy.data.sample import AIA_193_IMAGE


def main() -> None:
    # Load the sample data from Solar Dynamics Observatory (SDO)'s Atmospheric Imaging Assembly (AIA)
    # This is a solar image taken at 193 Angstroms wavelength, which shows the Sun's corona
    aia_map_masked = sunpy.map.Map(
        AIA_193_IMAGE,
    )
    aia_map_original = sunpy.map.Map(AIA_193_IMAGE)  # Original map for reference

    # Create mask for bright regions (e.g., active regions, solar flares)
    # Pixels with intensity < 10% of maximum will be masked out
    threshold: float = 0.10
    bright_regions_mask: npt.NDArray[np.bool_] = (
        aia_map_original.data < aia_map_original.max() * threshold
    )
    aia_map_masked.mask = bright_regions_mask

    # Plot masked solar image showing only bright regions
    fig = plt.figure()
    ax = fig.add_subplot(projection=aia_map_masked)
    aia_map_masked.plot(axes=ax)
    plt.colorbar()
    plt.show()

    # Apply Gaussian smoothing to reduce noise and connect nearby bright regions
    sigma: float = 14.0  # Width of the Gaussian kernel
    smoothed_data: npt.NDArray = ndimage.gaussian_filter(
        aia_map_original.data * ~bright_regions_mask,
        sigma,
    )
    smoothed_data[smoothed_data < 100] = 0  # Remove very faint regions

    # Create new map with smoothed solar data
    aia_map_smoothed = sunpy.map.Map(smoothed_data, aia_map_original.meta)

    # Label and count distinct bright regions in the solar image
    labels, region_count = ndimage.label(aia_map_smoothed.data)

    # Plot final result showing original image with contours around bright regions
    fig = plt.figure()
    ax = fig.add_subplot(projection=aia_map_original)
    aia_map_original.plot(axes=ax)
    ax.contour(labels)
    plt.figtext(0.3, 0.2, f"Number of bright regions = {region_count}", color="white")
    plt.show()


if __name__ == "__main__":
    main()
