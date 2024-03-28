import cv2
import numpy as np
import sys
def load():
    img_load = cv2.imread('pg.jpg')
    if img_load is None:
        sys.exit("Nie mozna wczytac obrazka")
    else:
        return img_load

def rgb_to_gray(rgb_img):
    r, g, b = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
    gray_img = 0.299 * r + 0.587 * g + 0.114 * b
    gray_img = gray_img.astype(np.uint8)
    return gray_img

def image_derivatives(image):
    img=[]
    image_height, image_width = image.shape
    for i in range(0, image_height):
        row = []
        for j in range(0, image_width):
            pixel = image.item(i, j)
            row.append(pixel)
        img.append(row)
    img=np.array(img)
    kernel_x=np.array([1,0,-1])
    kernel_y=np.array([[1],[0],[-1]])
    result_x = np.zeros_like(img)
    result_y = np.zeros_like(img)

    for i in range(1, image_height - 1):
        for j in range(1, image_width - 2):
            conv_x = img[i-1, j - 1] * kernel_x[0] + img[i-1, j] * kernel_x[1] + img[i-1, j + 1] * kernel_x[2]

            result_x[i, j] = conv_x

    for i in range(1, image_height - 2):
        for j in range(1, image_width - 1):
            conv_y = img[i - 1, j - 1] * kernel_y[0, 0] + img[i, j - 1] * kernel_y[1, 0] + img[i + 1, j - 1] * kernel_y[
                2, 0]

            result_y[i, j] = conv_y

    return result_x, result_y
def MatM(image, k=0.05):
    gray_img = rgb_to_gray(image)
    image_height, image_width = gray_img.shape
    matrix = np.zeros((image_height, image_width))
    Ix, Iy = image_derivatives(gray_img)
    Ix = (Ix - np.min(Ix)) / (np.max(Ix) - np.min(Ix))
    Iy = (Iy - np.min(Iy)) / (np.max(Iy) - np.min(Iy))
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    for i in range(image_height):
        for j in range(image_width):
            sum11 = Ix2[i, j]
            sum22 = Iy2[i, j]
            sum12 = Ixy[i, j]
            det = sum11 * sum22 - sum12 ** 2
            tr = sum11 + sum22
            R = det - k * (tr ** 2)
            matrix[i, j] = R
    matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    return matrix, image_width, image_height
def harris_corner_detector(image):
    matrix, image_width, image_height = MatM(image)
    threshold = np.max(matrix) * 0.98

    # Create a copy of the image to draw circles on
    result_image = np.copy(image)

    for i in range(0, image_height):
        for j in range(0, image_width):
            value = matrix[i, j]
            if value > threshold:
                # Manually draw a circle using NumPy operations
                radius = 3
                color = (255, 0, 0)  # Blue color for the circle
                center = (j, i)
                cv2.circle(result_image, center, radius, color)  # -1 fills the circle

    cv2.imwrite("corners.png", result_image)
    return result_image

img = load()
gray_img=rgb_to_gray(img)
# cv2.namedWindow("Obraz", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Obraz", 1000, 600)
# cv2.imshow("Obraz",gray_img)
# cv2.waitKey()
gradient_x,gradient_y = image_derivatives(gray_img)
gradient_x_display = cv2.convertScaleAbs(gradient_x)
gradient_y_display = cv2.convertScaleAbs(gradient_y)
cv2.imwrite('pox.png',gradient_x)
cv2.imwrite('poy.png',gradient_y)
corners = harris_corner_detector(img)