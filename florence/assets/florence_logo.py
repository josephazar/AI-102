import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

def create_florence_logo():
    """
    Create a simple Florence logo as a PIL Image
    Returns a BytesIO object containing the logo image
    """
    # Create a new image with a white background
    img = Image.new('RGB', (300, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw a blue rectangle as background for "FLORENCE"
    draw.rectangle([(20, 30), (280, 80)], fill=(0, 120, 212))
    
    # Try to use a font, or fall back to default
    try:
        font = ImageFont.truetype("Arial", 36)
        small_font = ImageFont.truetype("Arial", 20)
    except IOError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw the text
    draw.text((40, 35), "FLORENCE", fill=(255, 255, 255), font=font)
    draw.text((100, 95), "Vision AI", fill=(0, 120, 212), font=small_font)
    
    # Save to BytesIO
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    
    return buffer

def get_logo_path():
    """
    Returns a path to save the Florence logo
    """
    return "florence/assets/florence_logo.png"

def save_logo():
    """
    Save the Florence logo to the assets directory
    """
    buffer = create_florence_logo()
    path = get_logo_path()
    
    try:
        with open(path, "wb") as f:
            f.write(buffer.getvalue())
        return True
    except Exception as e:
        print(f"Error saving logo: {e}")
        return False

if __name__ == "__main__":
    save_logo()