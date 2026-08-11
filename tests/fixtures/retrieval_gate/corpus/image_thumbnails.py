"""Thumbnail generation with aspect ratio preservation and EXIF orientation."""

import logging

logger = logging.getLogger(__name__)

THUMBNAIL_SIZES = {"small": (128, 128), "medium": (320, 320), "large": (768, 768)}
JPEG_QUALITY = 82
EXIF_ORIENTATION_TAG = 274


def scaled_dimensions(width, height, bounds):
    """Return dimensions that fit inside bounds while preserving aspect ratio.

    Scaling by the smaller of the two ratios is what keeps the image inside the
    box on both axes; scaling by the larger overflows one of them.
    """
    max_width, max_height = bounds
    if width <= max_width and height <= max_height:
        return width, height
    ratio = min(max_width / width, max_height / height)
    return max(1, int(width * ratio)), max(1, int(height * ratio))


def normalise_orientation(image):
    """Rotate an image to match its EXIF orientation tag and drop the tag.

    Cameras record orientation as metadata rather than rotating the pixels, so a
    thumbnail generated without applying it comes out sideways even though the
    original looked upright in a viewer that honours the tag.
    """
    exif = getattr(image, "getexif", dict)()
    orientation = exif.get(EXIF_ORIENTATION_TAG)
    rotations = {3: 180, 6: 270, 8: 90}
    if orientation in rotations:
        return image.rotate(rotations[orientation], expand=True)
    return image


def generate_thumbnail(image, size_name="medium"):
    """Produce one thumbnail at a named size."""
    if size_name not in THUMBNAIL_SIZES:
        raise KeyError(f"unknown thumbnail size '{size_name}', expected one of {sorted(THUMBNAIL_SIZES)}")
    upright = normalise_orientation(image)
    target = scaled_dimensions(upright.width, upright.height, THUMBNAIL_SIZES[size_name])
    return upright.resize(target)


def generate_all(image):
    """Produce every configured thumbnail size from one source image."""
    upright = normalise_orientation(image)
    return {name: generate_thumbnail(upright, name) for name in THUMBNAIL_SIZES}
