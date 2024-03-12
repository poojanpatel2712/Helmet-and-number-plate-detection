import cv2
import matplotlib.pyplot as plt
import numpy as np
def predict1(image_path):
    plt.style.use('dark_background')
    # img_ori = cv2.imread("presentation\\np_without_helmet.jpeg")
    img_ori = cv2.imread("np_with_helmet.jpg")
    height, width, channel = img_ori.shape

    plt.figure(figsize=(12, 10))
    plt.imshow(img_ori, cmap='gray')
    plt.axis('off')
    plt.savefig('Car.png',bbox_inches = 'tight')
    # plt.show()

    # In[1]
    # Convert Image to Grayscale
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(12, 10))
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    plt.savefig('Car-GrayScale.png',bbox_inches = 'tight')
    # plt.show()

    # In[2]
    # # Maximize Contrast
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    plt.figure(figsize=(12, 10))
    plt.imshow(gray, cmap='gray')
    plt.axis('off')
    plt.savefig('Car-Contrast.png',bbox_inches = 'tight')
    # plt.show()

    # In[3]
    # # Adaptive Thresholding
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    img_thresh = cv2.adaptiveThreshold(
        img_blurred, 
        maxValue=255.0, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=19, 
        C=9
    )

    plt.figure(figsize=(12, 10))
    plt.imshow(img_thresh, cmap='gray')
    plt.axis('off')
    plt.savefig('Car-Adaptive-Thresholding.png',bbox_inches = 'tight')
    # plt.show()

    # In[4]
    # Finding Contours to locate plate
    contours, _= cv2.findContours(
        img_thresh, 
        mode=cv2.RETR_LIST, 
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

    plt.figure(figsize=(12, 10))
    plt.imshow(temp_result)
    plt.axis('off')
    plt.savefig('Car-Contours.png',bbox_inches = 'tight')
    # plt.show()

