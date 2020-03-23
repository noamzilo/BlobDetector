# Open University 22928 Computer Vision, Ex 1
BlobDetector, Laplacian of Gaussian, Harris, Canny

Find the full answers under Explanations.pdf

# Question 1:
d. Perform Canny edge detection, with different parameters.
e. Calculate Harris corners for 2 sets of parameters.

# Question 2 - Create a blob detector using Laplacian of Gaussian [LoG]:
1. Build pyramids:
    1. Create Gaussian filters with increasing scale.
    2. Create Pyramids with corresponding filters. Normalize by sigma^2.
2. Non maximum supression - Search for local maxima both in image space and in scale space.
3. Present results over various scales.

![](https://i.imgur.com/5D95iE6.jpg)
