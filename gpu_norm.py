import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchstain.torch.normalizers import TorchMacenkoNormalizer, TorchReinhardNormalizer, TorchMultiMacenkoNormalizer
from tqdm import tqdm
import concurrent.futures

class DeviceNormalizer:
    """
    Wrapper class that handles moving tensors to and from a specified device.
    
    This class enables GPU acceleration without modifying the original normalizer implementations.
    It works by:
    1. Moving input tensors to the GPU before processing
    2. Moving normalizer parameters (HERef, maxCRef) to the GPU
    3. Moving results back to CPU for further processing/saving
    
    The original normalizer code runs unmodified, but operations happen on the GPU.
    """
    
    def __init__(self, normalizer, device):
        """
        Initialize with a normalizer and a device.
        
        Args:
            normalizer: A normalizer instance (e.g., TorchMacenkoNormalizer)
            device: Device to use (e.g., 'cuda', 'cuda:0', 'cpu')
        """
        self.normalizer = normalizer
        self.device = device
        
        # Using PyTorch's automatic GPU acceleration by moving tensors to device
        self._move_params_to_device()
    
    def _move_params_to_device(self):
        """
        Move normalizer parameters to the specified device.
        
        This leverages PyTorch's ability to run operations on the device
        where the tensors are located. By moving these parameters to GPU,
        all operations involving them will execute on the GPU.
        """
        # Move reference matrices to device - these are key for speedup
        if hasattr(self.normalizer, 'HERef'):
            self.normalizer.HERef = self.normalizer.HERef.to(self.device)
        if hasattr(self.normalizer, 'maxCRef'):
            self.normalizer.maxCRef = self.normalizer.maxCRef.to(self.device)
            
        # For augmentors that store reference data
        if hasattr(self.normalizer, 'CRef'):
            self.normalizer.CRef = self.normalizer.CRef.to(self.device)
    
    def fit(self, I, **kwargs):
        """
        Fit the normalizer using an image on the device.
        
        Args:
            I: A single image tensor or a list of image tensors
            **kwargs: Additional arguments to pass to the normalizer's fit method
        """
        try:
            if isinstance(I, list):
                # For normalizers that accept a list of images (e.g., MultiMacenkoNormalizer)
                device_Is = [img.to(self.device) for img in I]
                self.normalizer.fit(device_Is, **kwargs)
            else:
                device_I = I.to(self.device)
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
            Normalized image tensor(s), moved back to CPU
        """
        try:
            # Move input to device
            device_I = I.to(self.device)
            
            # Call the normalizer's normalize method
            # The computation happens on the GPU because the inputs are on GPU
            result = self.normalizer.normalize(device_I, **kwargs)
            
            # Move results back to CPU for further processing
            if isinstance(result, tuple) and len(result) == 3:
                # Most normalizers return (normalized, H, E)
                normalized, H, E = result
                return (
                    normalized.cpu() if normalized is not None else None, 
                    H.cpu() if H is not None else None, 
                    E.cpu() if E is not None else None
                )
            else:
                # In case some normalizer returns a single tensor
                return result.cpu() if result is not None else None
        except Exception as e:
            print(f"Error during normalization: {e}")
            # Return original image moved back to CPU as fallback
            return I.cpu()

class BatchHENormalizer:
    """Class for batch processing of H&E stained images with GPU support."""
    
    def __init__(self, normalizer_type='macenko', device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
        """
        Initialize the batch normalizer.
        
        Args:
            normalizer_type: Type of normalizer to use ('macenko', 'reinhard', or 'multimacenko')
            device: Device to use for processing ('cuda', 'cuda:0', 'cpu', etc.)
            **kwargs: Additional arguments to pass to the normalizer constructor
        """
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize the normalizer based on type
        if normalizer_type.lower() == 'macenko':
            normalizer = TorchMacenkoNormalizer()
        elif normalizer_type.lower() == 'reinhard':
            normalizer = TorchReinhardNormalizer(method=kwargs.get('method', None))
        elif normalizer_type.lower() == 'multimacenko':
            normalizer = TorchMultiMacenkoNormalizer(norm_mode=kwargs.get('norm_mode', 'avg-post'))
        else:
            raise ValueError(f"Unsupported normalizer type: {normalizer_type}")
        
        # Wrap with device management
        self.normalizer = DeviceNormalizer(normalizer, device)
    
    def fit(self, target_images, **kwargs):
        """
        Fit the normalizer to target images.
        
        Args:
            target_images: A list of image tensors, a single image tensor, or a batch tensor
            **kwargs: Additional arguments to pass to the normalizer's fit method
        """
        if isinstance(target_images, list):
            self.normalizer.fit(target_images, **kwargs)
        elif target_images.dim() == 4:  # Batch tensor [B, C, H, W]
            # If a batch tensor is provided, extract individual images
            images = [target_images[i] for i in range(target_images.shape[0])]
            self.normalizer.fit(images, **kwargs)
        else:  # Single image tensor [C, H, W]
            self.normalizer.fit(target_images, **kwargs)
    
    def normalize_image(self, image, **kwargs):
        """
        Normalize a single image.
        
        Args:
            image: A tensor of shape [C, H, W]
            **kwargs: Additional arguments to pass to the normalizer's normalize method
            
        Returns:
            A normalized image tensor with shape [C, H, W]
        """
        # Ensure the input image has the correct format
        if image.dim() != 3 or image.shape[0] != 3:
            print(f"Warning: Input image has unexpected shape: {image.shape}, attempting to correct...")
            if image.dim() == 3 and image.shape[2] == 3:  # [H, W, C] format
                image = image.permute(2, 0, 1)  # Convert to [C, H, W]
        
        # Call the normalizer and get the result
        result = self.normalizer.normalize(image, **kwargs)
        
        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 3:
            normalized, _, _ = result
        else:
            normalized = result
            
        # Ensure the output has the correct format [C, H, W]
        if normalized.dim() == 3 and normalized.shape[0] != 3 and normalized.shape[2] == 3:
            normalized = normalized.permute(2, 0, 1)
            
        return normalized
    
    def normalize_batch(self, images, **kwargs):
        """
        Normalize a batch of images.
        
        Args:
            images: A batch tensor of shape [B, C, H, W] or a list of image tensors
            **kwargs: Additional arguments to pass to the normalizer's normalize method
            
        Returns:
            A list of normalized image tensors
        """
        # Prepare batch
        if isinstance(images, list):
            batch_images = images
        else:  # Assume it's a batch tensor
            batch_images = [images[i] for i in range(images.shape[0])]
        
        # Normalize each image in the batch
        normalized_images = []
        for img in batch_images:
            normalized = self.normalize_image(img, **kwargs)
            normalized_images.append(normalized)
        
        return normalized_images
    
    def process_folder(self, input_folder, output_folder=None, batch_size=4, num_workers=4, 
                       fit_target=None, **kwargs):
        """
        Process all images in a folder with the normalizer.
        
        Args:
            input_folder: Path to folder containing images to normalize
            output_folder: Path to save normalized images (if None, overwrites original)
            batch_size: Number of images to process at once
            num_workers: Number of parallel workers for loading/saving images
            fit_target: Path to target image for normalization reference (if None, fits to first batch)
            **kwargs: Additional parameters to pass to the normalizer
        """
        # Setup transform for image loading
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)  # Scale to [0, 255] range
        ])
        
        # First fit the normalizer if a target is provided
        if fit_target is not None:
            print(f"Fitting normalizer to target: {fit_target}")
            target_image = Image.open(fit_target).convert('RGB')
            # Define transform here to ensure consistency
            target_tensor = transform(target_image).byte()
            
            # Pass the target tensor directly to fit - the method will handle device conversion
            self.fit(target_tensor, **kwargs)
        
        # Create output folder if specified
        if output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
            print(f"Will save normalized images to: {output_folder}")
        else:
            print("Will overwrite original images with normalized versions")
        
        # Get all image files in the folder
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        print(f"Found {len(image_files)} images to process")
        print(f"Processing in batches of {batch_size} images")
        
        # Create a thread pool for saving that will run in the background
        all_save_futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as save_executor:
            # Process images in batches
            for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
                batch_files = image_files[i:i+batch_size]
                
                # Load images in parallel
                batch_images = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {executor.submit(self._load_image, os.path.join(input_folder, f), transform): f 
                            for f in batch_files}
                    for future in concurrent.futures.as_completed(futures):
                        file_name = futures[future]
                        try:
                            img_tensor = future.result()
                            if img_tensor is not None:
                                batch_images.append((img_tensor, file_name))
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
                
                # Submit save operations to run in background
                for normalized, file_name in normalized_batch:
                    future = save_executor.submit(
                        self._save_image,
                        normalized,
                        file_name,
                        output_folder if output_folder else input_folder
                    )
                    all_save_futures[future] = file_name
            
            # Now that all batches are processed, wait for all save operations to complete
            print("All normalization complete. Waiting for remaining save operations to finish...")
            for future in tqdm(concurrent.futures.as_completed(all_save_futures), 
                               total=len(all_save_futures), 
                               desc="Finishing save operations"):
                file_name = all_save_futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error saving {file_name}: {e}")
        
        print("All operations completed.")
    
    @staticmethod
    def _load_image(image_path, transform):
        """
        Load an image and apply transformation.
        
        Args:
            image_path: Path to the image file
            transform: Transformation to apply to the image
            
        Returns:
            A tensor representation of the image
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_tensor = transform(img).byte()
            return img_tensor
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    @staticmethod
    def _save_image(normalized, file_name, output_folder):
        # Ensure the normalized tensor has the correct dimensions [C,H,W] and 3 channels
        if normalized.dim() != 3 or normalized.shape[0] != 3:
            print(f"Warning: Normalized tensor for {file_name} has unexpected shape: {normalized.shape}")
            # Try to reshape if needed
            if normalized.shape[0] == 1 and normalized.dim() == 3:
                # Grayscale image - convert to 3 channels
                normalized = normalized.repeat(3, 1, 1)
            elif normalized.dim() > 3:
                # Too many dimensions - squeeze out extra dimensions
                normalized = normalized.squeeze()
            
        # Convert to PIL for saving, ensuring byte format and proper range [0,255]
        normalized = torch.clamp(normalized, 0, 255).byte()
        img = transforms.ToPILImage()(normalized)
        save_path = os.path.join(output_folder, file_name)
        img.save(save_path)

# Example usage
if __name__ == "__main__":
    # Initialize the batch normalizer with GPU support
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    normalizer = BatchHENormalizer(normalizer_type='macenko', device=device)
    
    # Process a folder of images
    normalizer.process_folder(
        input_folder='data/tiles',           # Folder containing images to normalize
        output_folder='data/normed_tiles',     # Folder to save normalized images (set to None to overwrite originals)
        batch_size=8,                          # Number of images to process at once
        num_workers=4,                         # Number of parallel workers for loading images
        fit_target='data/target.png',         # Target image for normalization reference
        Io=240,                                # Additional parameters for the normalizer
        alpha=1
    )