from tkinter import *
from tkinter import messagebox
from PIL import ImageTk,Image
from tkinter.filedialog import askopenfilename
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from grad_cam import get_main

a=Tk()
a.title("Brain Tumor Detector")
a.geometry("1200x600")
a.minsize(1200,600)
a.maxsize(1200,600)

#load trained model
loaded_model1 = load_model("Project_Saved_Models/brain_tumor_model.h5")

#perform denoising
def denoise(image):

    #denoising using Non-local mean algorithm
    out = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    return out

def prediction1():

    list_box.insert(1,"Loading Image")
    list_box.insert(2,"")
    list_box.insert(3,"Image Preprocessing")
    list_box.insert(4,"")
    list_box.insert(5,"Loading BT Detection Model")
    list_box.insert(6,"")
    list_box.insert(7,"Prediction")


    image = cv2.imread(path)
    image_resize=cv2.resize(image,(150,150))
    output=denoise(image_resize)
    img_yuv = cv2.cvtColor(output,cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    print(hist_eq.shape)
    out_image=np.expand_dims(hist_eq,axis=0)
    out_image=np.expand_dims(out_image,axis=0)/255
    print(out_image.shape)

    my_pred = loaded_model1.predict(out_image)
    print(my_pred)
    my_pred=np.argmax(my_pred,axis=1)
    print(my_pred)
    my_pred=my_pred[0]
    print(my_pred)

    if my_pred==0:
        print("Glioma Tumor")
        a="Glioma Tumor"
    elif my_pred==1:
        a="Meningioma Tumor"
        print("Meningioma Tumor")
    elif my_pred==2:
        a="No Tumor"
        print("No Tumor")
    elif my_pred==3:
        a="Pituitary Tumor"
        print("Pituitary Tumor")

    out_label.config(text= a)
    get_main(path)


def Check():
    global f
    f.pack_forget()

    f=Frame(a,bg="white")
    f.pack(side="top",fill="both",expand=True)


    
    global f1
    f1=Frame(f,bg="Lavender")
    f1.place(x=0,y=0,width=560,height=610)
    f1.config()
                   
    input_label=Label(f1,text="INPUT",font="arial 16",bg="lavender")
    input_label.pack(padx=0,pady=20)



    upload_pic_button=Button(f1,text="Upload Picture",command=Upload,bg="pink")
    upload_pic_button.place(x=240,y=100)
    global label
    label=Label(f1,bg="Lavender")


    global f2
    f2=Frame(f,bg="aquamarine")
    f2.place(x=800,y=0,width=400,height=690)
    f2.config(pady=20)
    
    result_label=Label(f2,text="RESULT",font="arial 16",bg="aquamarine")
    result_label.pack(padx=0,pady=0)

    global out_label
    out_label=Label(f2,text="",bg="aquamarine",font="arial 16")
    out_label.pack(pady=90)
    global out_label1
    out_label1=Label(f2,text="",bg="aquamarine",font="arial 16")
    out_label1.pack()
    global out_label2
    out_label2=Label(f2,text="",bg="aquamarine",font="arial 16")
    out_label2.pack()


    f3=Frame(f,bg="Salmon")
    f3.place(x=560,y=0,width=240,height=690)
    f3.config()

    name_label=Label(f3,text="Process",font="arial 14",bg="Salmon")
    name_label.pack(pady=20)

    global list_box
    list_box=Listbox(f3,height=12,width=31)
    list_box.pack()

    predict_button1=Button(f3,text="Predict",command=prediction1,bg="deepskyblue")
    predict_button1.pack(side="top",pady=10)


def Upload():
    global path
    label.config(image='')
    list_box.delete(0,END)
    out_label.config(text='')
    path=askopenfilename(title='Open a file',
                         initialdir='Test_Images',
                         filetypes=(("JPG","*.jpg"),("JPEG","*.jpeg"),("PNG","*.png")))
    print("Path : ",path)
    image=Image.open(path)
    global imagename
    imagename=ImageTk.PhotoImage(image.resize((300,300)))
    label.config(image=imagename)
    label.image=imagename
    # label.pack()
    label.place(x=140,y=210)
                  


def Home():
    global f
    f.pack_forget()
    
    f=Frame(a,bg="cornflower blue")
    f.pack(side="top",fill="both",expand=True)

    front_image = Image.open("Project_Extra/home.jpg")
    front_photo = ImageTk.PhotoImage(front_image.resize((1200,600), Image.ANTIALIAS))
    front_label = Label(f, image=front_photo)
    front_label.image = front_photo
    front_label.pack()

    home_label=Label(f,text="Brain Tumor Detector",font="arial 35",bg="white")
    home_label.place(x=300,y=290)




f=Frame(a,bg="cornflower blue")
f.pack(side="top",fill="both",expand=True)

front_image = Image.open("Project_Extra/home.jpg")
front_photo = ImageTk.PhotoImage(front_image.resize((1200,600), Image.ANTIALIAS))
front_label = Label(f, image=front_photo)
front_label.image = front_photo
front_label.pack()

home_label=Label(f,text="Brain Tumor Detector",font="arial 35",bg="white")
home_label.place(x=300,y=290)

m=Menu(a)
m.add_command(label="Home",command=Home)
checkmenu=Menu(m)
m.add_command(label="Check",command=Check)
a.config(menu=m)




a.mainloop()
