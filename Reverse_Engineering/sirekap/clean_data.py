import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
import csv
from skimage import measure
from skimage import morphology
import torchvision
import torchvision.transforms as transforms
from PIL import Image,ImageOps
from scipy.ndimage import center_of_mass, shift


def crop_and_resize(image, contour):

    x, y, w, h = cv2.boundingRect(max(contour, key=cv2.contourArea))
    aspect_ratio = w / float(h)  
    second_biggest_area=None
    if aspect_ratio > 5: 
         sorted_contour=sorted(contour, key=cv2.contourArea)
         second_biggest_area= sorted_contour[len(sorted_contour)-2]
         x,y,w,h=cv2.boundingRect(second_biggest_area)
    
    blurred = cv2.bilateralFilter(image,9,75,75)
    image=cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _,im_bw = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    
    image = 255 - im_bw
    digit = image[y:y+h, x:x+w]

    canvas = np.zeros((32, 32), dtype=np.uint8)

    
    if aspect_ratio > 1:  # Wide digit
        new_w = 20
        new_h = max(1,int(20 / aspect_ratio))
    else:  
        new_h = 20
        new_w = max(1,int(20 * aspect_ratio))


    print(new_w)
    print(new_h)
    print("Before resize")
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print("After resize resize")

    
    x_offset = (32 - new_w) // 2
    y_offset = (32 - new_h) // 2

    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized

    return canvas,second_biggest_area

def process_image_using_contours(image, k=0):
    
    
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, im_bw = cv2.threshold(imgray, 90, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    

    im_bw = 255 - im_bw


    contours, _ = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    processed_image = image.copy()
    print(len(contours))
    x,y,w,h=cv2.boundingRect(max(contours, key=cv2.contourArea))
    aspect_ratio = w / float(h)  # Calculate aspect ratio (width/height)
    if aspect_ratio > 5: #since width is very long in horizontal it must mean aspect ratio is larger than 5
         sorted_contour=sorted(contours, key=cv2.contourArea)
         second_biggest_area= sorted_contour[len(sorted_contour)-2]
         x,y,w,h=cv2.boundingRect(second_biggest_area)
        
    cv2.rectangle(processed_image,(x,y),(x+w,y+h),(0,255,0),1)

   

    return processed_image,contours


def transform_to_tensor(nd_image): #image already resized  from function crop and resize
        pil_image = Image.fromarray(nd_image)
        transform_image= transforms.Compose([  
        transforms.ToTensor(),       
    ])
        result = transform_image(pil_image)
        return result
def transform_image(image_cv2,cvPath):
    _,contour=process_image_using_contours(image_cv2)
 
    cropped_image,_=crop_and_resize(image_cv2,contour)
    kernel = np.full((1,1),5)
    dilated= cv2.dilate(cropped_image, kernel, iterations=1)
    edges = cv2.Laplacian(dilated, cv2.CV_64F)
    edges = cv2.convertScaleAbs(edges)  
    mean_brightness= np.mean(edges)
    print(mean_brightness)
    if mean_brightness<40:
        enhanced= cv2.addWeighted(dilated,1,edges,1,0)
        cropped_image=cv2.convertScaleAbs(enhanced, alpha=3, beta=0)
    return transform_to_tensor(cropped_image)
     
def detect_image_with_horizontal_line(aspect_ratio,cvPath):
    if aspect_ratio > 5:
           with open('horizontal.csv', 'a', newline='') as file:
                writer=csv.writer(file)
                writer.writerow([cvPath])

def image_processing_no_effect(image,image_path):
    image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_image = 255 - image 
    pil_image=Image.fromarray(inverted_image)
    transform= transforms.Compose([

    transforms.Resize((32, 32)),  
        transforms.ToTensor(),       
        transforms.Normalize((0.5,), (0.5,)),  
     
    ])
    result = transform(pil_image)
    return result


def image_processing_sirekap(image, image_path):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    thresh = cv2.adaptiveThreshold(
        resized, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11, 7
    )

    
    inverted = cv2.bitwise_not(thresh)


    pil_image = Image.fromarray(inverted)

    transform = transforms.Compose([
        transforms.ToTensor(),                
        transforms.Normalize((0.5,), (0.5,))  
    ])

    return transform(pil_image)  

def image_processing_sirekap_lenet(image, image_path):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    thresh = cv2.adaptiveThreshold(
        resized, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11, 7
    )
    inverted = cv2.bitwise_not(thresh)
    pil_image = Image.fromarray(inverted)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),                
        transforms.Normalize((0.5,), (0.5,))  
    ])

    return transform(pil_image) 
def preprocess_mnist(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    aspect_ratio = w / h

    if aspect_ratio > 1: 
        new_w = 20
        new_h = max(1, round(20 / aspect_ratio))
    else:  
        new_h = 20
        new_w = max(1, round(20 * aspect_ratio))

    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    new_image = np.zeros((28, 28), dtype=np.uint8)  
    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2
    new_image[start_y:start_y+new_h, start_x:start_x+new_w] = resized

    def shift_to_center(image):
        cy, cx = np.array(center_of_mass(image)) 
        if np.isnan(cy) or np.isnan(cx): 
            return image
        shift_x = round(14 - cx)
        shift_y = round(14 - cy)

        return shift(image, shift=[shift_y, shift_x], mode='nearest')

    centered_image = shift_to_center(new_image)
    centered_image = np.where(centered_image > 128, 255, 0).astype(np.uint8)

    return centered_image

def remove_artificial_white_border(image, border_size=5):

    h, w = image.shape

    
    image[:border_size, :] = 0  
    image[-border_size:, :] = 0 
    image[:, :border_size] = 0  
    image[:, -border_size:] = 0 

    return image
def transform_image_otsu_only(img,imgpath):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, otsu_image1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted_result = cv2.bitwise_not(otsu_image1)
    border_size=5
    otsu_colored= cv2.copyMakeBorder(inverted_result, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=255)
    contours, _ = cv2.findContours(otsu_colored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea) if contours else None
    cropped_image=otsu_colored
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = cropped_image[y:y+h,x:x+w]
    removed_border=remove_artificial_white_border(cropped_image)
    processed= preprocess_mnist(removed_border)
    pil_image = Image.fromarray(processed)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),                
        transforms.Normalize((0.5,), (0.5,))  
    ])
    tensor = transform(pil_image)
    print("this is tensor.shape",tensor.shape)

    return tensor 
    
    