import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def center_crop_image(image_path, target_size=224):
    """
    Center crop a 2000x2000 image to target_size x target_size
    
    Args:
        image_path (str): Path to the input JPEG image
        target_size (int): Size of the output square crop (default: 224)
    
    Returns:
        tuple: (original_image, cropped_image)
    """
    # Load the image using PIL
    original_img = Image.open(image_path)
    
    # Convert to numpy array for processing
    img_array = np.array(original_img)
    
    # Get image dimensions
    height, width = img_array.shape[:2]
    print(f"Original image size: {width}x{height}")
    
    # Calculate center crop coordinates
    center_x, center_y = width // 2, height // 2
    half_crop = target_size // 2
    
    # Define crop boundaries
    left = center_x - half_crop
    right = center_x + half_crop
    top = center_y - half_crop
    bottom = center_y + half_crop
    
    # Perform center crop
    cropped_img = img_array[top:bottom, left:right]
    
    print(f"Cropped image size: {cropped_img.shape[1]}x{cropped_img.shape[0]}")
    
    return img_array, cropped_img

def display_images(original, cropped, save_path=None):
    """
    Display original and cropped images side by side
    
    Args:
        original (np.array): Original image array
        cropped (np.array): Cropped image array
        save_path (str, optional): Path to save the comparison plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original image
    ax1.imshow(original)
    ax1.set_title(f'Original Image\n{original.shape[1]}x{original.shape[0]}', fontsize=12)
    ax1.axis('off')
    
    # Display cropped image
    ax2.imshow(cropped)
    ax2.set_title(f'Center Cropped Image\n{cropped.shape[1]}x{cropped.shape[0]}', fontsize=12)
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison saved to: {save_path}")
    
    plt.show()

def save_cropped_image(cropped_img, output_path):
    """
    Save the cropped image as JPEG
    
    Args:
        cropped_img (np.array): Cropped image array
        output_path (str): Path to save the cropped image
    """
    # Convert back to PIL Image and save
    cropped_pil = Image.fromarray(cropped_img)
    cropped_pil.save(output_path, 'JPEG', quality=95)
    print(f"Cropped image saved to: {output_path}")

# Main execution
if __name__ == "__main__":
    # Specify your image path here
    image_path = "/cs/student/projects1/aibh/2024/elnefary/data/raw/ECU/ECU_raw_6/328__20220327_154752.isyntax/Da1305.jpg"  # Replace with your actual image path
    
    try:
        # Perform center crop
        original, cropped = center_crop_image(image_path, target_size=224)
        
        # Display results
        display_images(original, cropped, save_path="image_comparison.png")
        
        # Save cropped image
        save_cropped_image(cropped, "cropped_224x224.jpg")
        
    except FileNotFoundError:
        print(f"Error: Could not find image at '{image_path}'")
        print("Please update the image_path variable with the correct path to your JPEG image")
    except Exception as e:
        print(f"Error processing image: {str(e)}")

# Alternative: Create a sample image for testing
def create_sample_image():
    """Create a sample 2000x2000 image for testing"""
    # Create a colorful gradient image
    img = np.zeros((2000, 2000, 3), dtype=np.uint8)
    
    # Create gradient patterns
    for i in range(2000):
        for j in range(2000):
            img[i, j, 0] = (i * 255) // 2000  # Red gradient
            img[i, j, 1] = (j * 255) // 2000  # Green gradient
            img[i, j, 2] = ((i + j) * 255) // 4000  # Blue gradient
    
    # Add some geometric shapes for visual reference
    center = 1000
    cv2.circle(img, (center, center), 300, (255, 255, 255), 5)
    cv2.rectangle(img, (center-200, center-200), (center+200, center+200), (0, 0, 0), 3)
    
    # Save sample image
    sample_path = "sample_2000x2000.jpg"
    cv2.imwrite(sample_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Sample image created: {sample_path}")
    
    return sample_path

# Uncomment the lines below to create and test with a sample image
sample_path = create_sample_image()
# original, cropped = center_crop_image(sample_path, target_size=224)
display_images(original, cropped)