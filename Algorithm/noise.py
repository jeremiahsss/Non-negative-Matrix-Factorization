import numpy as np


class Noise():
    def __init__(self, img = None, img_shape = (56, 46), noise_type = "block", b = 10, level = 0.2, ratio = 0.2, mean = 0, std = 10):
        """
        Args:
            img: Input array
            img_shape = Image shape (in 2D)
            noise_type = Name of the noise type
            b: Length of square block for block occulusion noise 
            level: Noise level for salt and pepper noise(0-1)
            ratio: Ratio of Salt vs Pepper for salt and pepper noise
            mean: Mean of the Gaussian noise
            std: Standard deviation of Gaussian noise
        """
        self.img = img
        self.img_shape = img_shape
        self.noise_type = noise_type
        self.b = b
        self.level = level
        self.ratio = ratio
        self.mean = mean
        self.std = std
    
    def fit(self):
        return self
    
    def generate_noise(self):
        """
        Returns:
            contaminated: Contaminated image 
            noise: Noise array
        """
        if self.noise_type == "block":
            contaminated, noise = Noise.block_occulusion_noise(self)
        elif self.noise_type == "saltpepper":
            contaminated, noise = Noise.salt_pepper_noise(self)
        elif self.noise_type == "gaussian":
            contaminated, noise = Noise.gaussian_noise(self)
        else:
            raise ValueError(f"Invalid noise type: {self.noise_type}")
        
        return contaminated, noise


    def block_occulusion_noise(self):
        """
        Returns:
            contaminated: Contaminated image with block occulusion noise
            noise: Block occulusion noise
        """
        # Create a zero-array with the same shape as the input image for noise tracking
        noise = np.zeros_like(self.img)

        # Create temporary array
        contaminated = np.zeros_like(self.img)

        for i in range(self.img_shape[1]):  
            # Randomly select the starting coordinate of the block
            x_coord = np.random.randint(0, self.img_shape[1] - self.b)
            y_coord = np.random.randint(0, self.img_shape[0] - self.b)

            # Create temporary arrays and reshape to 2D
            temp_img = np.copy(self.img)[:, i].reshape(self.img_shape)
            temp_noise = np.zeros_like(self.img)[:, i].reshape(self.img_shape)

            # Set selected block block area to white (255 or 1 after min-max normalisation)
            temp_img[y_coord:y_coord + self.b, x_coord:x_coord + self.b] = 1
            temp_noise[y_coord:y_coord + self.b, x_coord:x_coord + self.b] = 1

            # Flatten the image array
            contaminated[:, i] = temp_img.flatten()
            noise[:,i] = temp_noise.flatten()

        return contaminated, noise
    
    def salt_pepper_noise(self):
        """
        Returns:
            contaminated: Contaminated image with salt and pepper noise
            noise: salt and pepper noise
        """

        # Create temporary array
        contaminated = np.copy(self.img)

        # Create a temporary array with the same shape as the input image for noise tracking
        noise = np.copy(self.img)

        # Set all pixels in the noise array to gray (128)
        noise[:] = 0.5

        # Get the number of salt and pepper noise
        num_total = int(np.ceil(self.level * self.img[:,0].size))
        num_salt = int(np.ceil(num_total * self.ratio))

        for i in range(self.img.shape[1]):
            # Randomly select pixel indices for transformation
            total_indices = np.random.choice(self.img[:,0].size , num_total, replace = False)

            # Randomly select element from total_indices to be salt noise
            salt = np.random.choice(total_indices.size, num_salt, replace=False)

            # Select indices for salt noise
            salt_indices = total_indices[salt]

            # Select indices for pepper noise
            pepper_indices = np.delete(total_indices, salt)

            # Add salt noise
            contaminated[:,i][salt_indices] = 1
            noise[:,i][salt_indices] = 1

            # Add pepper noise
            contaminated[:,i][pepper_indices] = 0
            noise[:,i][pepper_indices] = 0
                    
        return contaminated, noise
     

    def gaussian_noise(self):
        """
        Returns:
            contaminated = Contaminated image with clipped Gaussian noise
            noise = Gaussian noise
        """
        # Create a zero-array with the same shape as the input image for noise tracking
        noise = np.zeros_like(self.img)

        # Generate gaussian noise
        gauss_noise = np.random.normal(self.mean, self.std, self.img.shape)

        # Add gaussian noise to the image array
        contaminated = self.img + gauss_noise

        # Clip the values to ensure they are within the range [0, 1]
        contaminated = np.clip(contaminated, 0, 1)

        # Calculate the actual noise added
        noise = contaminated - self.img

        return contaminated, noise
    

    
        
