import cv2 as cv
import tkinter as tk
from tkinter import *
from tkinter import filedialog as fd
root = tk.Tk()
root.title(	'shitty img manipulation')
root.geometry("850x700")	


v=tk.IntVar()
c=tk.IntVar()
d=tk.IntVar()

#functions

def inimg():
	file_path =fd.askopenfilename()
	img= cv.imread(file_path)
	if (v.get()==1):	
		cv.imshow('guko',img)
		cv.waitKey(0)
	elif(v.get()==2):
		dd=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		cv.imshow('black guko',dd)
		cv.waitKey(0)	



LabelFrame(root, text=" load image ", width=160, height=70).place(x=10,y=10)


input_img  = Button(root,text='add image',command=inimg).place(x=20 ,y = 40)

LabelFrame(root,text='convert',width=120 ,height=90).place(x = 200 , y=10)	
    
default = Radiobutton(root ,text= 'default',variable=v,value =1).place(x=200,y=40)

gray = Radiobutton(root ,text= 'gray color',variable=v,value =2 ).place(x=200,y=60)

LabelFrame(root,text='add noise ',width=180 ,height=100).place(x = 400 , y=10)

noise1 = Radiobutton(root ,text= 'salt & pepper',variable=c,value =1).place(x=400,y=40)
noise2 = Radiobutton(root ,text= 'gaussian',variable=c,value =2).place(x=400,y=60)
noise3 = Radiobutton(root ,text= 'poison',variable=c,value =3).place(x=400,y=80)
	
#point transform

LabelFrame(root,text='point transform ',width=250,height=160).place(x = 10 , y=100)

brightness = Button(root,text='brightness adj',padx=50).place(x=20,y=120)

contrast = Button(root,text='contrast adj',padx=57).place(x=20,y=150)

histogram = Button(root,text='histogram',padx=63).place(x=20,y=180)

histogram_eq = Button(root,text='histogram equalizer',padx=31).place(x=20,y=210)

#local transform 

LabelFrame(root,text='local transform ',width=800,height=150).place(x = 10 , y=260)

low_pass = Button(root,text='low pass filter',padx=30).place(x=20,y=280)
	
high_pass = Button(root,text='high pass filter',padx=28).place(x=20,y=310)

median = Button(root,text='median',padx = 52).place(x=20,y=340)

avg_filter = Button(root,text='averageing filter',padx=22).place(x=20,y=370)

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
line_detect = Button(root,text='line detect haugh transform',padx=20).place(x=20,y=460)
circule_detect = Button(root,text='circule detect haugh transform').place(x=20,y=500)

LabelFrame(root,text='morphological op\'s ',width=400,height=160).place(x = 350 , y=420)

dilation = Button(root,text='dilation',padx=30).place(x=360,y=450)
erosion = Button(root,text='erosion',padx=30).place(x=360,y=480)
close = Button(root,text='close',padx=38).place(x=360,y=510)
Open = Button(root,text='open',padx=40).place(x=360,y=540)

choices = ['Arbitrary', 'diamond', 'disk', 'line', 'octagon', 'pair', 'periodic','line', 'rectangle','square']
variable = StringVar(root)
variable.set('arbitrary')

Label(root,text='kernal',fg='green').place(x=550,y=440)
kernal= OptionMenu(root, variable, *choices,).place(x=550,y=470)

LabelFrame(root,width=400,height=50).place(x=160,y=589)
save =Button(root,text ='save',padx=50).place(x=200 ,y=600)
exit=Button(root,text='Exit',command=quit,padx=50).place(x=400 ,y=600)




root.mainloop()
