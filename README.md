Image quantization by using k-means. 

k-Means uses n amount of centroids to cluster data points into n amount of clusters. 

luckily each pixel in an image could be represented in a 3 dimentional matrix using three colors red, green and blue in (R,G,B). choosing n amount of clusters helps in reducing the size of the image but it reduces the accuracy of colors.
A normal picture has values from 0 to 255 for each of three colors metioned perior, so each take log2(256) = 8 bits, meaning a signle pixel is 24 bits, with k-means we could reduce the number of bits needed to represent an image.
