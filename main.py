# Image Segmentation based in color range

import cv2
import matplotlib.pyplot as plt
import numpy as np


# look at all the color space conversions OpenCV provides
flags = [i for i in dir(cv2) if i.startswith('COLOR')]

# ## NEMO image
nemo = cv2.imread('images/nemo0.jpg')

# examine an image
# OpenCV by default reads images in BGR format
# plt.imshow(nemo)
# plt.show()

# ## RGB NEMO
# use the cvtColor(image, flag) flag fix this
nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)

# ## HSV NEMO
hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)

plt.imshow(nemo)
# plt.show()

# ## Compare the image in both RGB and HSV color spaces
# ## by visualizing the color distribution of its pixels.
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

# Place each pixel in its location based on its components
# and color it by its color.


# ## RGB Plot

# 1. split the image and set up the 3D plot:
# OpenCV split() splits an image into its component channels.
# r,g,b = cv2.split(nemo)
# fig = plt.figure()
# axis = fig.add_subplot(1,1,1, projection='3d')

# 2. set up the pixel colors.
# Reshape and normalize pixels to color them with its true color :
# Flat the colors of pixels in the image into a list to normalize,
# then pass it to the facecolors parameter of Matplotlib scatter().
# ( Normalizing means condensing the range of colors from 0-255 to 0-1
# as required for the facecolors parameter.
# Lastly, facecolors wants a list, not an NumPy array. )
pixel_colors = nemo.reshape((np.shape(nemo)[0]*np.shape(nemo)[1],3))
norm = colors.Normalize(vmin=-1,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

# all the components ready for plotting:
# the pixel positions for each axis
# and their corresponding colors,
# in the format facecolors expect
# axis.scatter(r.flatten(),g.flatten(),b.flatten(), facecolors=pixel_colors,marker='.')
# axis.set_xlabel('Red')
# axis.set_ylabel('Green')
# axis.set_zlabel('Blu')


# ## HSV Plot
h,s,v = cv2.split(hsv_nemo)
fig = plt.figure()
axis = fig.add_subplot(1,1,1, projection='3d')

axis.scatter(h.flatten(),s.flatten(),v.flatten(), facecolors=pixel_colors,marker='.')
axis.set_xlabel('Hue')
axis.set_ylabel('Saturation')
axis.set_zlabel('Value')

# plt.show()



# ### Picking Out a Range
# threshold based on a simple range of oranges
# choose the range by eyeballing the plot above
# or using a color picking

light_orange = (1,190,200)
dark_orange = (18,255,255)

# Display the colors in Matplotlib
# Matplotlib only interprets colors in RGB,
# but are provided handy conversion functions
# to plot images for the major color spaces

from matplotlib.colors import hsv_to_rgb

# 10x10x3 colored squares
# using NumPy to easily fill them
lo_square = np.full((10,10,3), light_orange, dtype=np.uint8) / 255.0
do_square = np.full((10,10,3), dark_orange, dtype=np.uint8) / 255.0

plt.subplot(1,2,1)
plt.imshow(hsv_to_rgb(do_square))
plt.subplot(1,2,2)
plt.imshow(hsv_to_rgb(lo_square))
# plt.show()



# Once you get a decent color range,
# you can use cv2.inRange() to try to threshold Nemo.
# inRange() takes three parameters: the image, lower range, higher range.
# It returns a binary mask (an ndarray of 1s and 0s)
# the size of the image where values of 1 indicate values within the range,
# and zero values indicate values outside
mask = cv2.inRange(hsv_nemo, light_orange, dark_orange)


# To impose the mask on top of the original image, use cv2.bitwise_and(),
# which keeps every pixel in the given image if the corresponding value in the mask is 1:
result = cv2.bitwise_and(nemo,nemo,mask=mask)


# To see what that did exactly,
# let’s view both the mask and the original image with the mask on top:
plt.subplot(1,2,1)
plt.imshow(mask,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(result)
# plt.show()


# A decent job of capturing the orange parts of the fish.
# The only problem is that Nemo also has white stripes…

# Adding a second mask that looks for whites
# is very similar to what you did already with the oranges:

# specifie a color range
light_white = (0,0,200)
dark_white = (145,60,255)

# To display the whites,
# you can take the same approach as we did previously with the oranges:
lw_square = np.full((10,10,3), light_white, dtype=np.uint8) / 255.0
dw_square = np.full((10,10,3), dark_white, dtype=np.uint8) / 255.0

plt.subplot(1,2,1)
plt.imshow(hsv_to_rgb(lw_square))
plt.subplot(1,2,2)
plt.imshow(hsv_to_rgb(dw_square))
# plt.show()



# The upper range here is a very blue white,
# because the white does have tinges of blue in the shadows.
# A second mask to see if it captures Nemo’s stripes
# the same way as you did the first:
mask_white = cv2.inRange(hsv_nemo, light_white, dark_white)
result_white = cv2.bitwise_and(nemo,nemo,mask=mask_white)

plt.subplot(1,2,1)
plt.imshow(mask_white,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(result_white)
# plt.show()



# Now combine the masks.
# Add masks together that results in 1 values
# wherever there is orange or white and plot the results:
final_mask = mask + mask_white

final_result = cv2.bitwise_and(nemo,nemo,mask=final_mask)
plt.subplot(1,2,1)
plt.imshow(final_mask,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(final_result)
# plt.show()







# Essentially, you have a rough segmentation of Nemo in HSV color space.
# You’ll notice there are a few stray pixels along the segmentation border,
# and if you like, you can use a Gaussian blur to tidy up the small false detections.

# A Gaussian blur is an image filter
# that uses a kind of function to transform each pixel in the image
# smoothing out image noise and reducing detail.
# Here’s what applying the blur looks like for our image:
blur = cv2.GaussianBlur(final_result, (7,7),0)
plt.imshow(blur)
# plt.show()









# ## Does This Segmentation Generalize to Nemo’s Relatives?
# let’s see how well this segmentation technique generalizes to other clownfish images.

# follow in segmentation.py file














#
