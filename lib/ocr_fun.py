import easyocr
#import pytesseract
import cv2
import numpy as np
import os
import re 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
#import keras_ocr



class ocr:
    def __init__(self) -> None:
        pass
#    def pytesseract_fun( self , img) :
#        # you can add here some preprocessing 
#        norm_img = np.zeros((img.shape[0], img.shape[1]))
#        imgn = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
#        imgn = cv2.threshold(imgn, 100, 255, cv2.THRESH_BINARY)[1]
#        imgn = cv2.GaussianBlur(imgn, (1, 1), 0)
#        img_rgb = cv2.cvtColor(imgn, cv2.COLOR_BGR2RGB)
#        result = pytesseract.image_to_string(img_rgb)
#        return result

    def easyocr_fun(self , img) : 
        # you can add here some preprocessing 
        #img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        reader = easyocr.Reader(['en'])
        result = reader.readtext(gray)
        numbers = re.findall(r'\d+', result[0][-2])
        numbers = "".join(numbers)
        return numbers
    def detect_numbers(img):
        # Load the image
        #image = cv2.imread(image_path)

        # Initialize the EasyOCR reader
        reader = easyocr.Reader(['en'])

        # Perform OCR on the image
        result = reader.readtext(img)

        # Extract numbers from the OCR result
        numbers = []
        for detection in result:
            text = detection[1]
            if text.isdigit():
                numbers.append(int(text))

        return numbers
    #def keras_ocr_fun(self , img) :
    #    pipeline = keras_ocr.pipeline.Pipeline() 
    #    imgr = keras_ocr.tools.read(img)
    #    results = pipeline.recognize(imgr)
    #    return results
    def AnPr(self , img) :
        img = cv2.imread(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, np.uint8)
        (x,y) = np.where(mask==255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        text = result[0][-2]
        return text