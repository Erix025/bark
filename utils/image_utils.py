from PIL import Image

def resize_image(image : Image.Image, height, width):
    rate = min(height / image.size[1], width / image.size[0])
    new_size = (int(image.size[0] * rate), int(image.size[1] * rate))
    # pad the image to the target size
    image = image.resize(new_size)
    new_image = Image.new("RGB", (width, height))
    new_image.paste(image, ((width - new_size[0]) // 2, (height - new_size[1]) // 2))
    return new_image
    