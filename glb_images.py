"""Keep a glb's JPEG textures JPEG when trimesh writes them back out.

trimesh loads glTF images as PIL images without a ``format``, and its exporter writes any
image that is not marked JPEG as PNG: the 8192^2 JPEG textures of a 5M-face source (2.5 MB
each) became 23 MB PNGs and cost ~10 s each to encode, once per export of parts.glb,
boxes.glb and the completed parts. Standard library only, so every venv can import it.
"""
import json
import struct

TEXTURE_KEYS = ("image", "baseColorTexture", "metallicRoughnessTexture", "normalTexture",
                "emissiveTexture", "occlusionTexture")


def glb_image_mimes(path):
    """Set of image mime types declared in a .glb's JSON chunk (empty for anything else)."""
    try:
        with open(path, "rb") as f:
            header = f.read(20)
            if len(header) < 20 or header[:4] != b"glTF":
                return set()
            length = struct.unpack("<I", header[12:16])[0]
            tree = json.loads(f.read(length))
    except (OSError, ValueError):
        return set()
    return {image.get("mimeType") for image in tree.get("images", [])}


def keep_jpeg(materials, source_glb):
    """Mark the textures of `materials` JPEG when every image in `source_glb` is JPEG.

    Returns the number of images marked. Mixed or PNG sources are left alone.
    """
    if glb_image_mimes(source_glb) != {"image/jpeg"}:
        return 0
    marked = 0
    for material in materials:
        for key in TEXTURE_KEYS:
            image = getattr(material, key, None)
            if image is not None and hasattr(image, "format") and image.format is None:
                image.format = "JPEG"
                marked += 1
    return marked


def mesh_materials(geometry):
    """Materials of trimesh geometries (a Trimesh, a Scene, or an iterable of them)."""
    if hasattr(geometry, "geometry") and isinstance(geometry.geometry, dict):
        geometry = geometry.geometry.values()
    elif not isinstance(geometry, (list, tuple)) and not hasattr(geometry, "__next__"):
        geometry = [geometry]
    found = []
    for geom in geometry:
        material = getattr(getattr(geom, "visual", None), "material", None)
        if material is not None and all(material is not m for m in found):
            found.append(material)
    return found
