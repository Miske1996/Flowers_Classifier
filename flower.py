from fastai import *
from fastai.vision import *
from fastai.vision.image import pil2tensor,Image
import cv2
import numpy as np
import os
learn = load_learner('./')

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):	
	bg_img = background_img.copy()	
	if overlay_size is not None:
		img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)
	# Extract the alpha mask of the RGBA image, convert to RGB 
	b,g,r,a = cv2.split(img_to_overlay_t)
	overlay_color = cv2.merge((b,g,r))	
	# Apply some simple filtering to remove edge noise
	mask = cv2.medianBlur(a,5)
	h, w, _ = overlay_color.shape
	roi = bg_img[y:y+h, x:x+w]
	# Black-out the area behind the logo in our original ROI
	img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))	
	# Mask out the logo from the logo image.
	img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)
	# Update the original image with our new ROI
	bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)
	return bg_img
for i in os.listdir('flowers'):
    flower_files=[]
    folder_flower = i   
    for c in os.listdir('flowers/'+i):
        flower_files.append('flowers/'+i+'/'+c)

    first_half = flower_files[0:(int(len(flower_files)/2))]
    p = len(flower_files)
    second_half = flower_files[int(p/2):p]
    for c in range(int(p/2)):
        img1 = cv2.imread(first_half[c])
        img2 = cv2.imread(second_half[c])
        size = (400,400)
        img1 = cv2.resize(img1,size)
        img2 = cv2.resize(img2,size)
        img= np.concatenate((img1,img2),axis=1)
        img1_pred = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        img2_pred = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
        pred_1 = str(learn.predict(Image(pil2tensor(img1_pred,np.float32).div_(255)))[0])
        pred_2 = str(learn.predict(Image(pil2tensor(img2_pred,np.float32).div_(255)))[0])
        print(pred_1,i)
        if str(pred_1) == str(i):
            src = cv2.imread('true.png', -1)
            img = overlay_transparent(img, src, 150, 100, (150,150))
        else:
            src = cv2.imread('wrong.png', -1)
            img = overlay_transparent(img, src, 150, 100, (150,150))

        if str(pred_2) == str(i):
            src = cv2.imread('true.png',-1)
            img = overlay_transparent(img, src, 530, 100, (150,150))
        else:
            src = cv2.imread('wrong.png', -1)
            img = overlay_transparent(img, src, 530, 100, (150,150))   
        cv2.putText(img,'prediction :'+pred_1,(0,380),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.9,(0,255,255),1)
        cv2.putText(img,'prediction :'+pred_2,(400,380),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.9,(0,255,255),1)
        cv2.putText(img,'Categorie : '+i,(0,30),cv2.FONT_HERSHEY_PLAIN,2.5,(0,255,255),3)
        # cv2.imshow('Flowers',img)
        cv2.imshow('image',img)
        k = cv2.waitKey(30)&0xff
        if k == ord('q'):
            quit()
cv2.destroyAllWindows()