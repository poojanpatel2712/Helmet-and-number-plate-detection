import cv2
import matplotlib.pyplot as plt
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
