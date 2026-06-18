"""Generate test fixture assets for pipeline tests."""

import numpy as np
from PIL import Image
from pathlib import Path


def generate_test_image(path: str, width: int = 256, height: int = 256) -> Path:
    """Generate a test image with a colored shape on a white background."""
    arr = np.ones((height, width, 4), dtype=np.uint8) * 255  # white background

    # Draw a red circle
    cy, cx = height // 2, width // 2
    radius = min(height, width) // 4
    y, x = np.ogrid[:height, :width]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    arr[mask] = [255, 50, 50, 255]  # red

    # Draw a blue square offset
    sq_size = width // 6
    x_start, y_start = width // 2 + sq_size, height // 2 - sq_size
    arr[y_start : y_start + sq_size, x_start : x_start + sq_size] = [50, 50, 255, 255]

    # Draw a green triangle
    for i in range(sq_size):
        for j in range(i + 1):
            arr[height // 2 + i, width // 4 + j] = [50, 255, 50, 255]

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "RGBA").save(out)
    return out


def generate_test_mesh(path: str, num_vertices: int = 100) -> Path:
    """Generate a simple OBJ mesh (sphere-ish) for testing."""
    import math

    lines = ["# AssetForge test mesh", "o TestMesh"]
    vertices = []
    faces = []

    # Generate a simple shape
    for i in range(num_vertices):
        theta = 2 * math.pi * i / num_vertices
        r = 1.0 + 0.3 * math.sin(theta * 3)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = 0.2 * math.sin(theta * 4)
        vertices.append((x, y, z))
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")

    # Generate edges/triangles
    for i in range(num_vertices - 1):
        j = (i + 1) % num_vertices
        # Create a triangle strip pattern
        lines.append(f"f {i + 1} {j + 1} {(j + 2) % num_vertices + 1}")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def generate_scene_image(path: str, width: int = 128, height: int = 128) -> Path:
    """Generate a 'scene' image with multiple colored shapes (for photo set)."""
    arr = np.ones((height, width, 3), dtype=np.uint8) * 200  # light gray bg

    # Simple "object" — a colored rectangle with a triangle on top
    arr[40:100, 30:90] = [100, 150, 200]  # rectangle
    for i in range(30):
        for j in range(i, 60 - i):
            if 20 + i < height and 30 + j < width:
                arr[20 + i, 30 + j] = [100, 200, 100]  # triangle

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "RGB").save(out)
    return out


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    print("Generating test fixtures...")
    generate_test_image(str(base / "fixtures" / "test_input.png"))
    generate_test_mesh(str(base / "fixtures" / "test_mesh.obj"))
    print(f"  fixtures/test_input.png")
    print(f"  fixtures/test_mesh.obj")

    photo_dir = base.parent / "examples" / "sample_photo_set"
    photo_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        angle = i * 30  # different rotation
        generate_scene_image(str(photo_dir / f"view_{i:02d}.jpg"))
        print(f"  examples/sample_photo_set/view_{i:02d}.jpg")
    print("Done.")
