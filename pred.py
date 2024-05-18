import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

#load trained model
loaded_model = load_model("Project_Saved_Models/brain_tumor_model.h5")

#perform denoising
def denoise(image):

    #denoising using Non-local mean algorithm
    out = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    return out

def main1(path):

    image = cv2.imread(path)
    image_resize=cv2.resize(image,(150,150))
    output=denoise(image_resize)
    img_yuv = cv2.cvtColor(output,cv2.COLOR_BGR2YUV)
    # apply histogram equalization 
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    hist_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    print(hist_eq.shape)
    out_image=np.expand_dims(hist_eq,axis=0)
    out_image=np.expand_dims(out_image,axis=0)/255
    print(out_image.shape)


    my_pred = loaded_model.predict(out_image)
    print(my_pred)
    my_pred=np.argmax(my_pred,axis=1)
    print(my_pred)

    if my_pred==0:
        print("Glioma Tumor")
    elif my_pred==1:
        print("Meningioma Tumor")
    elif my_pred==2:
        print("No Tumor")
    elif my_pred==3:
        print("Pituitary Tumor")



if __name__=="__main__":
    from tkinter.filedialog import askopenfilename
    path=askopenfilename()
    main1(path)
