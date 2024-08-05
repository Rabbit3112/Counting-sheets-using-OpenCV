import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Function to perform noise reduction
def reduce_noise(image):
    # Convert image to a NumPy array
    image_np = np.array(image)
    
    # Apply Bilateral Filter for noise reduction
    denoised_image = cv2.bilateralFilter(image_np, 9, 75, 75)
    
    return denoised_image

# Function to sharpen the image
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1,  5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

# Function to blur the background
def blur_background(image, mask):
    blurred_background = cv2.GaussianBlur(image, (51, 51), 0)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(blurred_background, blurred_background, mask=mask_inv)
    result += cv2.bitwise_and(image, image, mask=mask)
    return result

# Function to count sheets in an image with enhanced processing
def count_sheets(image):
    # Convert PIL image to a NumPy array
    image = np.array(image.convert('RGB'))
    
    # Reduce noise
    denoised_image = reduce_noise(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(denoised_image, cv2.COLOR_RGB2GRAY)
    
    # Sharpen the grayscale image
    sharpened = sharpen_image(gray)
    
    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
    
    # Create a mask for the foreground
    mask = cv2.threshold(adaptive_thresh, 160, 255, cv2.THRESH_BINARY)[1]
    
    # Blur the background
    blurred_image = blur_background(image, mask)
    
    # Convert to grayscale again for edge detection
    gray_blurred = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2GRAY)
    
    # Use Canny edge detection
    edges = cv2.Canny(gray_blurred, 51, 51)
    
    # Find contours to detect sheets
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Prepare an output image to draw the contours
    output_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Initialize a list to store the coordinates of the detected sheets
    num_sheets = 0
    
    for contour in contours:
        # Filter out small contours
        if cv2.contourArea(contour) > 50:  # Adjust size threshold as needed
            # Draw contour on the output image
            cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)
            num_sheets += 1
    
    # Display intermediate images for debugging
    debug_image('Gray Image', gray)
    debug_image('Sharpened Image', sharpened)
    debug_image('Adaptive Threshold Image', adaptive_thresh)
    debug_image('Blurred Background Image', blurred_image)
    debug_image('Edges Image', edges)
    
    return num_sheets, output_image

def debug_image(title, image):
    st.image(image, caption=title, use_column_width=True)

# Streamlit UI
st.balloons()
st.title("SheetStack: An Intelligent Sheet Counting Application")
st.write(" ")
st.markdown(""" This project is a web application that counts the number of sheets in an image using OpenCV and Streamlit. It processes the image through sharpening, adaptive thresholding, and edge detection to identify and count the sheets. """)
st.write(" ")
st.markdown("""The application displays the results with annotated contours and provides a user-friendly interface for image upload and visualization. It effectively separates the foreground (sheets) from the background for accurate detection. The tool is designed for easy use and precise sheet counting.""")
st.write(" ")
st.write(" ")
st.subheader("PROJECT WORKFLOW")
st.write(" ")
st.image("D:\\Projects\\Counting-papers-using-OpenCV\\image\\WORKFLOW.png")
st.write(" ")
st.write(" ")
st.title("How it works?")
  
st.subheader("What is computer vision?")
st.write(" ")
st.write("""Computer vision leverages 
artificial intelligence (AI)
 to allow computers to obtain meaningful data from visual inputs such as photos and videos. The insights gained from computer vision are then used to take automated actions. Just like AI gives computers the ability to ‘think’, computer vision allows them to ‘see’.""")
st.write(" ")
st.image("D:\\Projects\\Counting-papers-using-OpenCV\\image\\com vision.png")
st.write(" ")
st.write(" ")
st.write("""Two key technologies drive computer vision: a convolutional neural network and deep learning, a type of machine learning.

Machine learning (ML) leverages algorithm-based models to enable computers to learn context through visual data analysis. Once sufficient data is provided to the model, it will be able to ‘see the big picture’ and differentiate between visual inputs. Instead of being programmed to recognize and differentiate between images, the machine uses AI algorithms to learn autonomously.

Convolutional neural networks help ML models see by fractionating images into pixels. Each pixel is given a label or tag. These labels are then collectively used to carry out convolutions, a mathematical process that combines two functions to produce a third function. Through this process, convolutional neural networks can process visual inputs.""")
st.write(" ")
st.write("""Examples of computer vision include: facial recognition, object detection, image segmentation, optical character recognition (OCR), and autonomous vehicle navigation.""")
st.write(" ")
st.write(" ")
st.write(" ")
st.image("D:\\Projects\\Counting-papers-using-OpenCV\\image\\eye ai.png")
st.write(" ")
st.write(" ")

st.subheader("A Web Application for Accurate Sheet Counting Using Image Processing")
st.write(" ")
st.write("""1. **Image Upload**: Users upload an image of sheet stacks through the Streamlit interface.
2. **Preprocessing**: The uploaded image is converted to grayscale and sharpened to enhance details.
3. **Foreground Segmentation**: Adaptive thresholding and contour detection are applied to separate sheets from the background.
4. **Background Blurring**: The background is blurred while keeping the foreground sheets sharp.
5. **Sheet Counting and Visualization**: Contours are detected, counted, and drawn on the image, with the total number of sheets displayed.""")
st.write(" ")
st.write(" ")
st.write(" ")

st.subheader(" Output image for Example")
st.write(" ")
st.write(" ")
st.image("D:\\Projects\\Counting-papers-using-OpenCV\\image\\example image.jpg")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.title('To use this application:')
st.image("D:\\Projects\\Counting-papers-using-OpenCV\\image\\D:\\Projects\\Corpus-Chat-bot\\image\\upload button.png")
st.write(" ")
st.write(" ")
st.subheader("Upload an image of sheet stacks, and the application will count the number of sheets.")
st.write(" ")
# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

count_placeholder = st.empty()
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_file)
    num_sheets, output_image = count_sheets(image)
    # Update the placeholder with the counted result
    count_placeholder.write(f"Number of sheets detected: {num_sheets}")
    st.subheader(f"Number of sheets detected: {num_sheets}")
    st.image(output_image, caption="Detected Sheets", use_column_width=True)
