import os
import tensorflow as tf
import numpy as np
from PIL import Image
import concurrent.futures
from tqdm import tqdm
import glob

class TFDeviceNormalizer:
    """
    Wrapper class that handles moving tensors to and from a specified device in TensorFlow.
    
    This class enables GPU acceleration without modifying the original normalizer implementations.
    It works by:
    1. Moving input tensors to the GPU 
    2. Setting TensorFlow to use the appropriate device
    3. Moving results back to CPU for further processing/saving
    """
    
    def __init__(self, normalizer, device):
        """
        Initialize with a normalizer and a device.
        
        Args:
            normalizer: A normalizer instance (e.g., TensorFlowMacenkoNormalizer)
            device: Device to use (e.g., '/GPU:0', '/CPU:0')
        """
        self.normalizer = normalizer
        self.device = device
        
        # Move attributes that are tensors to the device
        self._move_params_to_device()
    
    def _move_params_to_device(self):
        """
        Move normalizer parameters to the specified device.
        
        For TensorFlow, we need to explicitly recreate tensors on the target device.
        """
        with tf.device(self.device):
            # Move reference matrices to device if they exist
            if hasattr(self.normalizer, 'HERef') and isinstance(self.normalizer.HERef, tf.Tensor):
                self.normalizer.HERef = tf.identity(self.normalizer.HERef)
                
            if hasattr(self.normalizer, 'maxCRef') and isinstance(self.normalizer.maxCRef, tf.Tensor):
                self.normalizer.maxCRef = tf.identity(self.normalizer.maxCRef)
                
            # Handle other TensorFlow-specific attributes
            if hasattr(self.normalizer, 'CRef') and isinstance(self.normalizer.CRef, tf.Tensor):
                self.normalizer.CRef = tf.identity(self.normalizer.CRef)
    
    def fit(self, I, **kwargs):
        """
        Fit the normalizer using an image on the device.
        
        Args:
            I: A single image tensor or a list of image tensors
            **kwargs: Additional arguments to pass to the normalizer's fit method
        """
        try:
            with tf.device(self.device):
                if isinstance(I, list):
                    # For normalizers that accept multiple images
                    device_Is = [tf.identity(img) for img in I]
                    self.normalizer.fit(device_Is, **kwargs)
                else:
                    # Single image case
                    device_I = tf.identity(I)
                    self.normalizer.fit(device_I, **kwargs)
                
                # Re-move parameters to device after fitting
                self._move_params_to_device()
        except Exception as e:
            print(f"Error during fit: {e}")
            raise
    
    def normalize(self, I, **kwargs):
        """
        Normalize an image on the device.
        
        Args:
            I: An image tensor
            **kwargs: Additional arguments to pass to the normalizer's normalize method
            
        Returns:
            Normalized image tensor(s), moved back to CPU if needed
        """
        try:
            with tf.device(self.device):
                # Move input to device
                device_I = tf.identity(I)
                
                # Call the normalizer's normalize method
                result = self.normalizer.normalize(device_I, **kwargs)
                
            # TensorFlow automatically handles device placement for outputs,
            # but we'll ensure they're accessible from CPU
            if isinstance(result, tuple) and len(result) == 3:
                # Most normalizers return (normalized, H, E)
                normalized, H, E = result
                return (
                    normalized if normalized is not None else None,
                    H if H is not None else None,
                    E if E is not None else None
                )
            else:
                # In case some normalizer returns a single tensor
                return result
                
        except Exception as e:
            print(f"Error during normalization: {e}")
            # Return original image as fallback
            return I

class TFBatchHENormalizer:
    """Class for batch processing of H&E stained images with TensorFlow GPU support."""
    
    def __init__(self, normalizer_type='macenko', device='/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0', **kwargs):
        """
        Initialize the batch normalizer.
        
        Args:
            normalizer_type: Type of normalizer to use ('macenko', 'reinhard')
            device: Device to use for processing ('/GPU:0', '/CPU:0', etc.)
            **kwargs: Additional arguments to pass to the normalizer constructor
        """
        self.device = device
        print(f"Using device: {self.device}")
        
        # Set memory growth to avoid OOM errors
        self._configure_gpu_memory()
        
        # Import normalizers directly - don't use the factory function with 'backend' parameter
        from torchstain.tf.normalizers import TensorFlowMacenkoNormalizer, TensorFlowReinhardNormalizer

        # Initialize the normalizer based on type
        if normalizer_type.lower() == 'macenko':
            normalizer = TensorFlowMacenkoNormalizer()  # No backend parameter here
        elif normalizer_type.lower() == 'reinhard':
            normalizer = TensorFlowReinhardNormalizer(method=kwargs.get('method', None))  # Only pass method parameter
        else:
            raise ValueError(f"Unsupported normalizer type for TensorFlow: {normalizer_type}")
        
        # Wrap with device management
        self.normalizer = TFDeviceNormalizer(normalizer, device)
    
    def _configure_gpu_memory(self):
        """Configure TensorFlow to use memory growth for GPUs to avoid OOM errors."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("Memory growth enabled for GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(f"Error setting memory growth: {e}")
    
    def fit(self, target_images, **kwargs):
        """
        Fit the normalizer to target images.
        
        Args:
            target_images: A list of image tensors, a single image tensor, or a batch tensor
            **kwargs: Additional arguments to pass to the normalizer's fit method
        """
        if isinstance(target_images, list):
            self.normalizer.fit(target_images, **kwargs)
        elif len(tf.shape(target_images)) == 4:  # Batch tensor [B, H, W, C]
            # Extract individual images from batch
            batch_size = tf.shape(target_images)[0]
            images = [target_images[i] for i in range(batch_size)]
            self.normalizer.fit(images, **kwargs)
        else:  # Single image tensor [H, W, C]
            self.normalizer.fit(target_images, **kwargs)
    
    def normalize_image(self, image, **kwargs):
        """
        Normalize a single image.
        
        Args:
            image: A tensor of shape [H, W, C] or [C, H, W] (will be converted)
            **kwargs: Additional arguments to pass to the normalizer's normalize method
            
        Returns:
            A normalized image tensor with shape [H, W, C]
        """
        # Check if the tensor format needs conversion from PyTorch format
        if len(tf.shape(image)) == 3 and tf.shape(image)[0] == 3:  # [C, H, W] format (PyTorch)
            # TensorFlow's TensorFlowMacenkoNormalizer expects [C, H, W] format!
            # No need to transpose, as the TensorFlow normalizers are written to handle [C, H, W]
            pass
        elif len(tf.shape(image)) != 3:
            print(f"Warning: Input image has unexpected shape: {tf.shape(image)}")
        
        # Call the normalizer and get the result
        result = self.normalizer.normalize(image, **kwargs)
        
        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 3:
            normalized, _, _ = result
        else:
            normalized = result
            
        return normalized
    
    def normalize_batch(self, images, **kwargs):
        """
        Normalize a batch of images.
        
        Args:
            images: A batch tensor of shape [B, H, W, C] or a list of image tensors
            **kwargs: Additional arguments to pass to the normalizer's normalize method
            
        Returns:
            A list of normalized image tensors
        """
        # Prepare batch
        if isinstance(images, list):
            batch_images = images
        else:  # Assume it's a batch tensor
            batch_size = tf.shape(images)[0]
            batch_images = [images[i] for i in range(batch_size)]
        
        # Normalize each image in the batch
        normalized_images = []
        for img in batch_images:
            normalized = self.normalize_image(img, **kwargs)
            normalized_images.append(normalized)
        
        return normalized_images
    
    def process_folder(self, input_folder, output_folder=None, batch_size=4, num_workers=4, 
                       fit_target=None, **kwargs):
        """
        Process all images in a folder.
        
        Args:
            input_folder: Path to the folder containing images to normalize
            output_folder: Path to save the normalized images. If None, overwrite original images.
            batch_size: Number of images to process at once
            num_workers: Number of parallel workers for loading images
            fit_target: Path to a target image or folder of target images for fitting
            **kwargs: Additional arguments to pass to the normalizer's methods
        """
        # Create output folder if specified
        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
            print(f"Will save normalized images to: {output_folder}")
        else:
            print("Will overwrite original images with normalized versions")
        
        # Fit the normalizer if a target is provided
        if fit_target is not None:
            if os.path.isdir(fit_target):
                # Fit using all images in the target folder
                target_files = glob.glob(os.path.join(fit_target, "*.png")) + \
                               glob.glob(os.path.join(fit_target, "*.jpg")) + \
                               glob.glob(os.path.join(fit_target, "*.jpeg")) + \
                               glob.glob(os.path.join(fit_target, "*.tif")) + \
                               glob.glob(os.path.join(fit_target, "*.tiff"))
                
                target_images = []
                
                print(f"Loading {len(target_files)} target images for fitting...")
                for f in target_files:
                    img_tensor = self._load_tf_image(f)
                    if img_tensor is not None:
                        target_images.append(img_tensor)
                
                print("Fitting normalizer to target images...")
                self.fit(target_images, **kwargs)
            else:
                # Fit using a single target image
                print(f"Loading target image for fitting: {fit_target}")
                img_tensor = self._load_tf_image(fit_target)
                if img_tensor is not None:
                    self.fit(img_tensor, **kwargs)
        
        # Get all image files in the folder
        image_files = glob.glob(os.path.join(input_folder, "*.png")) + \
                      glob.glob(os.path.join(input_folder, "*.jpg")) + \
                      glob.glob(os.path.join(input_folder, "*.jpeg")) + \
                      glob.glob(os.path.join(input_folder, "*.tif")) + \
                      glob.glob(os.path.join(input_folder, "*.tiff"))
        
        print(f"Found {len(image_files)} images to process")
        print(f"Processing in batches of {batch_size} images")
        
        # Process images in batches
        for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
            batch_files = image_files[i:i+batch_size]
            
            # Load images in parallel
            batch_images = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(self._load_tf_image, f): f for f in batch_files}
                for future in concurrent.futures.as_completed(futures):
                    file_name = futures[future]
                    try:
                        img_tensor = future.result()
                        if img_tensor is not None:
                            batch_images.append((img_tensor, os.path.basename(file_name)))
                    except Exception as e:
                        print(f"Error loading {file_name}: {e}")
            
            # Normalize the batch
            normalized_batch = []
            for img_tensor, file_name in batch_images:
                try:
                    normalized = self.normalize_image(img_tensor, **kwargs)
                    normalized_batch.append((normalized, file_name))
                except Exception as e:
                    print(f"Error normalizing {file_name}: {e}")
            
            # Save results
            for normalized, file_name in normalized_batch:
                try:
                    # Ensure normalized is a proper image format
                    if tf.is_tensor(normalized):
                        # Convert to numpy for saving with PIL
                        normalized_np = normalized.numpy()
                    else:
                        normalized_np = np.array(normalized)
                    
                    # Ensure proper shape and data type for PIL
                    if normalized_np.ndim == 3:
                        # If output is [H, W, C] format, keep as is
                        if normalized_np.shape[-1] == 3:
                            pass
                        # If output is [C, H, W] format, convert to [H, W, C]
                        elif normalized_np.shape[0] == 3:
                            normalized_np = np.transpose(normalized_np, (1, 2, 0))
                    
                    # Ensure values are in valid range for images
                    normalized_np = np.clip(normalized_np, 0, 255).astype(np.uint8)
                    
                    # Save the image
                    img = Image.fromarray(normalized_np)
                    save_path = os.path.join(output_folder if output_folder else input_folder, file_name)
                    img.save(save_path)
                    
                except Exception as e:
                    print(f"Error saving {file_name}: {e}")
                    print(f"Shape of normalized tensor: {normalized_np.shape if 'normalized_np' in locals() else 'unknown'}")
    
    @staticmethod
    def _load_tf_image(image_path):
        """
        Load an image and convert to PyTorch-compatible format [C, H, W].
        
        Args:
            image_path: Path to the image file
            
        Returns:
            A TensorFlow tensor representing the image with shape [C, H, W]
        """
        try:
            # Read the image file
            img_data = tf.io.read_file(image_path)
            
            # Decode the image
            img_tensor = tf.image.decode_image(img_data, channels=3, expand_animations=False)
            img_tensor = tf.ensure_shape(img_tensor, [None, None, 3])
            
            # Convert from [H, W, C] to [C, H, W] for compatibility with TensorFlowMacenkoNormalizer
            # which expects the same format as PyTorch tensors
            img_tensor = tf.transpose(img_tensor, [2, 0, 1])
            
            return img_tensor
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Check for available GPU
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))
    
    # Initialize the batch normalizer with GPU support
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
    normalizer = TFBatchHENormalizer(normalizer_type='macenko', device=device)
    
    # Process a folder of images
    normalizer.process_folder(
        input_folder='data/tiles',                # Folder containing images to normalize
        output_folder='data/normed_tiles_tf',     # Folder to save normalized images (set to None to overwrite originals)
        batch_size=8,                             # Number of images to process at once
        num_workers=4,                            # Number of parallel workers for loading images
        fit_target='data/target.png',             # Target image for normalization reference
        Io=240,                                   # Additional parameters for the normalizer
        alpha=1
    )