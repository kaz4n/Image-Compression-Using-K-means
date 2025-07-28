import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from PIL import Image  # For loading/saving images

def compress_image(image_path, K=16, max_iter=100):
    # Load image and convert to numpy array
    img = Image.open(image_path)
    img = np.array(img) / 255.0  # Normalize to [0,1]

    # (width*height which is the number of pixels, 3) 
    h, w, d = img.shape
    pixel_data = img.reshape(-1, d)


    # run the k-means on smaller group of pixels to save time and computional power 
    # sampled_data = shuffle(pixel_data, random_state=0)[:5000]

    # kmeans = KMeans(n_clusters=K, max_iter=max_iter, n_init=10, random_state=0)
    # kmeans.fit(sampled_data)


    kmeans = KMeans(n_clusters=K, max_iter=max_iter, n_init=10, random_state=0)
    kmeans.fit(pixel_data)

    # Assign each pixel to centroid
    labels = kmeans.predict(pixel_data)
    compressed_data = kmeans.cluster_centers_[labels]

    # Reconstruct compressed image
    compressed_img = compressed_data.reshape(h, w, d)
    return compressed_img


original_path = "IMG_3468.jpg"
k = 128
compressed_img = compress_image(original_path, K=k, max_iter=100) 

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(plt.imread(original_path))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Compressed (K={k})")
plt.imshow(compressed_img)
plt.axis('off')
plt.show()