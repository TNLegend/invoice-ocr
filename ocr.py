import cv2
import pytesseract
from PIL import Image
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
def bw(image):
    return cv2.threshold(image, 200, 210, cv2.THRESH_BINARY)
def getBoxes(image,im):
    #processing the image in order to find the contours
    blur = cv2.GaussianBlur(image,(7,7),cv2.BORDER_DEFAULT)
    cv2.imwrite("temp/blur_image.jpg",blur)
    thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imwrite("temp/thresh_image.jpg", thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(8,10))
    cv2.imwrite("temp/kernel_image.jpg", kernel)
    dilate = cv2.dilate(thresh,kernel,iterations=1)
    cv2.imwrite("temp/dialated_image.jpg", dilate)
    cnts = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #finding the right contours
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #organize contour content from top to bottom
    cnts = sorted(cnts,key=lambda x: cv2.boundingRect(x)[1])
    #drawing contours
    ocr_list=list()
    for c in cnts:
        x,y,width,height = cv2.boundingRect(c)
        if width >50:

            roi = image[y:y+height,x:x+width]
            cv2.rectangle(im,(x,y),(x+width,y+height),(0,255,0),2)
            #recognizing the content of each box
            ocr_result = pytesseract.image_to_string(roi)
            ocr_list.append(ocr_result)
            print(ocr_list)
    cv2.imwrite("temp/boxes_image.jpg",im)
invoice_im = "./train/Invoice_38.png"
im = cv2.imread(invoice_im)
gray_image = grayscale(im)
thresh, im_bw = bw(gray_image)
cv2.imwrite("temp/bw_image.jpg", im_bw)
#trying to ocr image without determing boundaries => bad OCR result
print ("ocr before boxes identification : ")
img = Image.open("temp/bw_image.jpg")
ocr_result = pytesseract.image_to_string(img)
print(ocr_result)
#determing boundaries
print ("ocr after determining boundaries : ")
img = cv2.imread("temp/bw_image.jpg", cv2.IMREAD_GRAYSCALE)
getBoxes(img,im)
