import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def comprehensive_image_analysis(image_path):
    """
    Perform comprehensive image transformation and analysis
    
    Parameters:
    - image_path: Path to input image
    
    Returns:
    - Detailed transformation analysis results
    """
    # Read the original image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Basic Image Characteristics
    def get_image_characteristics(image):
        return {
            'Dimensions': image.shape,
            'Mean Pixel Intensity': np.mean(image),
            'Standard Deviation': np.std(image),
            'Min Pixel Value': np.min(image),
            'Max Pixel Value': np.max(image)
        }
    
    # Transformation Functions
    def translate_image(image, x_shift=50, y_shift=30):
        # Translation matrix
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    def rotate_image(image, angle=45):
        # Get image center
        center = (image.shape[1] // 2, image.shape[0] // 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    def scale_image(image, scale_factor=1.5):
        # Scaling
        return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, 
                          interpolation=cv2.INTER_LINEAR)
    
    # Image Quality Metrics
    def calculate_image_metrics(original, transformed):
        # Ensure same dimensions for comparison
        h, w = min(original.shape[0], transformed.shape[0]), \
               min(original.shape[1], transformed.shape[1])
        original = original[:h, :w]
        transformed = transformed[:h, :w]
        
        # Mean Squared Error
        mse = np.mean((original - transformed) ** 2)
        
        # Peak Signal-to-Noise Ratio
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
        
        # Structural Similarity Index
        ssim_score = ssim(original, transformed)
        
        return {
            'Mean Squared Error': mse,
            'Peak Signal-to-Noise Ratio (dB)': psnr,
            'Structural Similarity Index': ssim_score,
            'Information Preservation (%)': ssim_score * 100
        }
    
    # Perform Transformations
    translated_image = translate_image(original_image)
    rotated_image = rotate_image(original_image)
    scaled_image = scale_image(original_image)
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.title('Original Image')
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title('Translated Image')
    plt.imshow(translated_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title('Rotated Image')
    plt.imshow(rotated_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.title('Scaled Image')
    plt.imshow(scaled_image, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Comprehensive Analysis
    analysis_results = {
        'Original Image Characteristics': get_image_characteristics(original_image),
        'Translation Metrics': calculate_image_metrics(original_image, translated_image),
        'Rotation Metrics': calculate_image_metrics(original_image, rotated_image),
        'Scaling Metrics': calculate_image_metrics(original_image, scaled_image)
    }
    
    # Print Detailed Analysis
    for category, metrics in analysis_results.items():
        print(f"\n{category}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
    
    return analysis_results

# Example Usage
image_path = '/home/pratik/Downloads/ca2/gradient-shape-pattern.jpg'  # Replace with your image path
analysis_results = comprehensive_image_analysis(image_path)
