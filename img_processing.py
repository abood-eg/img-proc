import cv2 as cv
import tkinter as tk
import numpy as np
import random
from tkinter import *
from tkinter import filedialog as fd
from matplotlib import pyplot as plt
import math
from scipy import ndimage
root = tk.Tk()
root.title(	'shitty img manipulation')
root.geometry("850x700")	


v=tk.IntVar()
c=tk.IntVar()
d=tk.IntVar()

#functions
### salt and pepper noise 
def salt_pepper(img):
  
    # Getting the dimensions of the image
    row , col = img.shape
      
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
          
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
          
        # Color that pixel to white
        img[y_coord][x_coord] = 255
          
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(300 , 10000)
    for i in range(number_of_pixels):
        
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
          
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
          
        # Color that pixel to black
        img[y_coord][x_coord] = 0
          
    return img
### adding gaussian noise to the image 
def mat2gray(img):
    A = np.double(img)
    out = np.zeros(A.shape, np.double)
    normalized = cv.normalize(A, out, 1.0, 0.0, cv.NORM_MINMAX)
    return out
 
 #Add noise to the image
def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
    image = mat2gray(image)
    
    mode = mode.lower()
    if image.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
    if seed is not None:
        np.random.seed(seed=seed)
        
    if mode == 'gaussian':
        noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,
                                 image.shape)        
        out = image  + noise
    if clip:        
        out = np.clip(out, low_clip, 1.0)
        
    return out
def inimg():
	global file_path
	file_path =fd.askopenfilename()
	img= cv.imread(file_path)
	if (v.get()==1):	
		cv.imshow('default',img)
		cv.waitKey(0)
	elif(v.get()==2):
		dd=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		cv.imshow('gray',dd)
		cv.waitKey(0)
		cv.imwrite('gray-img.png',dd)
	if (c.get()==1):
	    img = cv.imread(file_path,cv.IMREAD_GRAYSCALE)
	    sp_img=salt_pepper(img)
	    cv.imshow('salt & pepper',sp_img)
	    cv.waitKey(0)
	    cv.imwrite('salt & pepper.png',sp_img)
	elif (c.get()==2):
		img = cv.imread(file_path,0)
		img1 = random_noise(img,'gaussian', mean=0.1,var=0.01)
		img1 = np.uint8(img1*255)
		cv.imshow('gaussian', img1)
		cv.waitKey(0)
		cv.destroyAllWindows()	
		cv.imwrite('gaussian.png',img1)
	elif (c.get()==3):
		img = (cv.imread(file_path)).astype(float)
		noise_mask = np.random.poisson(img) 
		noisy_img = img + noise_mask
		cv.imshow('poisson',noisy_img)
		cv.waitKey(0)
		cv.imwrite('poisson.png',noisy_img)
   
        		
## increase the brightness 
def bright():
	image = cv.imread(file_path)
	image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

	increase = 100

	v = image[:, :, 2]
	v = np.where(v <= 255 - increase, v + increase, 255)
	image[:, :, 2] = v

	image = cv.cvtColor(image, cv.COLOR_HSV2BGR)

	cv.imshow('Brightness', image)
	cv.waitKey(0)
	cv.destroyAllWindows()
	cv.imwrite('brightness.png',image)
### contrast adj 
def contrast_adj():
	img = cv.imread(file_path)
	imghsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
	imghsv[:,:,2] = [[max(pixel - 25, 0) if pixel < 190 else min(pixel + 25, 255) for pixel in row] for row in imghsv[:,:,2]]
	cv.imshow('contrast', cv.cvtColor(imghsv, cv.COLOR_HSV2BGR))
	cv.waitKey(0)
	cv.imwrite("contrast.png", cv.cvtColor(imghsv, cv.COLOR_HSV2BGR))
## now the histogram of the photo
def cv_histogram():
	img = cv.imread(file_path,0)
	plt.hist(img.ravel(),256,[0,256])
	plt.show()
## the histogram equalizer 
def histo_eq():
	img = cv.imread(file_path,0)
	hist,bins = np.histogram(img.flatten(),256,[0,256])
	cdf = hist.cumsum()
	cdf_normalized = cdf * float(hist.max()) / cdf.max()
	plt.plot(cdf_normalized, color = 'b')
	plt.hist(img.flatten(),256,[0,256], color = 'r')
	plt.xlim([0,256])
	plt.legend(('cdf','histogram'), loc = 'upper left')
	plt.show()
	cv.imshow('hest_eq',img)
	cv.waitKey(0)


### low pass filter 
def lo_pass():
	img = cv.imread(file_path, 0)
# Create a 5x5 Kernel
	kernel = np.ones((5,5), np.float32)/25
# Apply convolution between image and 5x5 Kernel
	dst = cv.filter2D(img,-1, kernel)
# Store LPF image as lpf.jpg
	cv.imshow("lpf.jpg", dst)
	cv.waitKey(0)
	cv.imwrite('lpdf.png',dst)

###avg filter
def avg():

# Read MyPic.jpg from system as grayscale
	img = cv.imread(file_path, 0)
# Apply Averaging blur
	blur = cv.blur(img,(5,5))
	cv.imshow("AvgBlur.jpg", blur)
	cv.waitKey(0)
	cv.imshow('avg.png',blur)
### median filter
def median_filter():
	img=cv.imread(file_path,0)
	img = cv.medianBlur(img,9)
	cv.imshow('median',img)
	cv.waitKey(0)
	cv.imwrite('median.png',img)

### high pass filter
def hi_pass():
	
	img = cv.imread(file_path,0)
	img =img - cv.GaussianBlur(img, (0,0), 3) + 127

	cv.imshow('highpass', img)
	cv.waitKey(0)	
	cv.imwrite('highpass.png',img)

#### erosion and dilation
def erode():
	img = cv.imread(file_path,0)
	kernel = np.ones((5,5),np.uint8)
	erosion = cv.erode(img,kernel,iterations = 1)
	cv.imshow('erosion',erosion)
	cv.waitKey(0)
	cv.imwrite('erosion.png',erosion)
def dilate():
	img = cv.imread(file_path,0)
	kernel = np.ones((5,5),np.uint8)
	dilation = cv.dilate(img,kernel,iterations = 1)
	cv.imshow('dilation',dilation)
	cv.waitKey(0)	
	cv.imwrite('dilation.png',dilation)

### opening and closing

def openin():
	img = cv.imread(file_path,0)
	kernel = np.ones((5,5),np.uint8)
	opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
	cv.imshow('opening',opening)
	cv.waitKey(0)	
	cv.imwrite('opening.png',opening)
def closin():
	img = cv.imread(file_path,0)
	kernel = np.ones((5,5),np.uint8)
	closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)	
	cv.imshow('closing',closing)
	cv.waitKey(0)
	cv.imwrite('closing.png',closing)

### line detction and circle detection
def line_detection():
	img = cv.imread(file_path)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	edges = cv.Canny(gray, 50, 150, apertureSize=3)
	cv.imshow("image", edges)
	cv.waitKey(0)
	minLineLength = 100
	maxLineGap = 10
	lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength, maxLineGap)
	for line in lines:
	    for x1, y1, x2, y2 in line:
	        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
	cv.imshow("line detected",img)
	cv.waitKey(0)
	cv.imwrite('line detection.png',img)
def circle_detection():

	img = cv.imread(file_path,0)
	img = cv.medianBlur(img,5)
	cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
	circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
	circles = np.uint16(np.around(circles))
	for i in circles[0,:]:
	    # draw the outer circle
	    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
	    # draw the center of the circle
	    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
	cv.imshow('detected circles',cimg)
	cv.waitKey(0)
	cv.destroyAllWindows()
	cv.imwrite("detected circles.jpg",cimg)


LabelFrame(root, text=" load image ", width=160, height=70).place(x=10,y=10)


input_img  = Button(root,text='add image',command=inimg).place(x=20 ,y = 40)

LabelFrame(root,text='convert',width=120 ,height=90).place(x = 200 , y=10)	
    
default = Radiobutton(root ,text= 'default',variable=v,value =1).place(x=200,y=40)

gray = Radiobutton(root ,text= 'gray color',variable=v,value =2 ).place(x=200,y=60)

LabelFrame(root,text='add noise ',width=180 ,height=100).place(x = 400 , y=10)

noise1 = Radiobutton(root ,text= 'salt & pepper',variable=c,value =1).place(x=400,y=40)
noise2 = Radiobutton(root ,text= 'gaussian',variable=c,value =2).place(x=400,y=60)
noise3 = Radiobutton(root ,text= 'poisson',variable=c,value =3).place(x=400,y=80)
	
#point transform

LabelFrame(root,text='point transform ',width=250,height=160).place(x = 10 , y=100)

brightness = Button(root,text='brightness adj',padx=50,command=bright).place(x=20,y=120)

contrast = Button(root,text='contrast adj',padx=57,command=contrast_adj).place(x=20,y=150)

histogram = Button(root,text='histogram',padx=63,command=cv_histogram).place(x=20,y=180)

histogram_eq = Button(root,text='histogram equalizer',padx=31,command=histo_eq).place(x=20,y=210)

#local transform 

LabelFrame(root,text='local transform ',width=800,height=150).place(x = 10 , y=260)

low_pass = Button(root,text='low pass filter',padx=30,command=lo_pass).place(x=20,y=280)
	
high_pass = Button(root,text='high pass filter',padx=28,command=hi_pass).place(x=20,y=310)

median = Button(root,text='median',padx = 52,command=median_filter).place(x=20,y=340)

avg_filter = Button(root,text='averageing filter',padx=22,command=avg).place(x=20,y=370)

LabelFrame(root,text='edge detection filters ',width=590,height=100).place(x = 200 , y=280)	

edge1 = Radiobutton(root ,text= 'lablacian',variable=d,value =1).place(x=200,y=300)
edge2 = Radiobutton(root ,text= 'gaussian',variable=d,value =2).place(x=200,y=320)
edge3 = Radiobutton(root ,text= 'vert.sobel',variable=d,value =3).place(x=200,y=340)
edge12 = Radiobutton(root ,text= 'horis.sobel',variable=d,value =4).place(x=350,y=300)
edge4 = Radiobutton(root ,text= 'vert.prewit',variable=d,value =5).place(x=350,y=320)
edge5 = Radiobutton(root ,text= 'horis.prewit',variable=d,value =6).place(x=350,y=340)
edge6 = Radiobutton(root ,text= 'lab of gaws',variable=d,value =7).place(x=500,y=300)
edge7 = Radiobutton(root ,text= 'canny ',variable=d,value =8).place(x=500,y=320)
edge8 = Radiobutton(root ,text= 'zero cross',variable=d,value =9).place(x=500,y=340)
edge9 = Radiobutton(root ,text= 'thiken',variable=d,value =10).place(x=650,y=300)
edge10 = Radiobutton(root ,text= 'skelton',variable=d,value =11).place(x=650,y=320)
edge11 = Radiobutton(root ,text= 'thinening',variable=d,value =12).place(x=650,y=340)


#global transform
LabelFrame(root,text='global transform ',width=300,height=140).place(x = 10 , y=420)	
line_detect = Button(root,text='line detect haugh transform',padx=20,command=line_detection).place(x=20,y=460)
circle_detect = Button(root,text='circle detect haugh transform',command=circle_detection).place(x=20,y=500)

LabelFrame(root,text='morphological op\'s ',width=400,height=160).place(x = 350 , y=420)

dilation = Button(root,text='dilation',padx=30,command=dilate).place(x=360,y=450)
erosion = Button(root,text='erosion',padx=30,command=erode).place(x=360,y=480)
close = Button(root,text='close',padx=38,command=closin).place(x=360,y=510)
Open = Button(root,text='open',padx=40,command=openin).place(x=360,y=540)

choices = ['Arbitrary', 'diamond', 'disk', 'line', 'octagon', 'pair', 'periodic','line', 'rectangle','square']
variable = StringVar(root)
variable.set('arbitrary')

Label(root,text='kernal',fg='green').place(x=550,y=440)
kernal= OptionMenu(root, variable, *choices,).place(x=550,y=470)

LabelFrame(root,width=440,height=50).place(x=160,y=589)
exit=Button(root,text='Exit',command=quit,padx=50).place(x=300 ,y=600)




root.mainloop()
