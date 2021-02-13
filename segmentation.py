import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb
import streamlit as st



st.title(Image Segmantation)
st.write("""
Image segmentation by color ranges with OpenCV. 
""")


# Combine all the preview code used to segment a single fish into a function
# that will take an image as input and return the segmented image.
# Expand this section to see what that looks like:


# load images into a list
path = 'images/nemo'
nemos_friends = []
for i in range(6):
    friend = cv2.cvtColor(cv2.imread(path + str(i) + '.jpg'), cv2.COLOR_BGR2RGB)
    nemos_friends.append(friend)




# Combine all the preview code into a function that will take an image as input
# and return the segmented image to segment a single fish

def segment_fish(image):
    ''' Attempts to segment the clownfish out of the provided image '''

    # convert the image into HSV
    hsv_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # Set the orange range
    light_orange = (1,190,200)
    dark_orange = (18,255,255)

    # Apply the orange mask
    mask = cv2.inRange(hsv_image,light_orange,dark_orange)

    # Set a white range
    light_white = (0,0,200)
    dark_white = (145,60,255)

    # Apply the mask
    mask_white = cv2.inRange(hsv_image, light_white, dark_white)

    # Combine the two masks
    final_mask = mask + mask_white
    result = cv2.bitwise_and(image,image,mask=final_mask)

    # Clean up the segmentation using blur
    # blur = cv2.GaussianBlur(result, (7,7),0)
    # return blur

    return result



# With that useful function, you can segment all the fish from a list
results = [segment_fish(friend) for friend in nemos_friends]

# Let’s view all the results by plotting them in a loop:
for i in range(1,6):
    # streamlit line
    fig,ax = plt.subplots()
    plt.subplot(1,2,1)
    plt.imshow(nemos_friends[i])
    plt.subplot(1,2,2)
    plt.imshow(results[i])
    # plt.show()
    # streamlit line
    st.pyplot(fig)






########################################################################
########################################################################
##################### Improve to image input ###########################
#### Unfinished: ColorPicker and Inputs to min and max color range #####
########################################################################


# streamlit adaptation
# import streamlit as st
# from bokeh.models import ColorPicker
# from bokeh.models.callbacks import CustomJS
# from PIL import ImageColor



# color_picker = ColorPicker(color="#ff4466", title="Choose color:", width=200)
# st.bokeh_chart(color_picker)
# # print(type(color_picker))
# st.write(color_picker.color)



# color_picker = ColorPicker(title="Choose color:")
# st.bokeh_chart(color_picker)
#
# user_input = st.text_input("HEX Code here:")
# if user_input:
#     st.write(user_input)



# def showplot(file_up):
#     fig,ax = plt.subplots()
#     plt.subplot(1,2,1)
#     plt.imshow(file_up)
#     st.pyplot(fig)


# uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg','jpg'])
# if uploaded_file is not None:
#     results = segment_fish(friend)
#     showplot(results)

















#
