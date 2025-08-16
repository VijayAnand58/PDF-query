from PIL import Image, ImageEnhance, ImageFilter


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

    # Resize if needed (some APIs limit image size)
    image = image.resize((512, 512))

    # Sharpen the image (especially for diagrams or charts)
    sharpener = ImageEnhance.Sharpness(image)
    image = sharpener.enhance(2.0)  # 1.0 = original, >1 = sharper

    # Optional: increase contrast for better visibility
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)

    return image
