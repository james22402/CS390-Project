import cv2
import numpy as np
import pytesseract as tes

#######################################################
#Get a somewhat cleaned image to work with. Base image#
#         to work with will be named "out"            #
#######################################################
oimg = cv2.imread('./Checks/real_check.jpg', cv2.IMREAD_COLOR)
scale_percent = 60 # percent of original size
width = int(oimg.shape[1] * scale_percent / 100)
height = int(oimg.shape[0] * scale_percent / 100) 
img = cv2.resize(oimg, (width,height))
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
imcontours=imgray
cv2.drawContours(imcontours, contours, -1, (0,255,0), 3)
cv2.imwrite('Temp\\\\1contours.jpg',imcontours)
max = 0
result = None
for i in range(0,len(contours)):
    if max < cv2.contourArea(contours[i]):
        max = cv2.contourArea(contours[i])
        result = contours[i]
mask = np.zeros_like(img)
cv2.drawContours(mask, [result], 0, (255,255,255), -1)
out = np.zeros_like(img)
out[mask == 255] = img[mask == 255]
cv2.imwrite('Temp\\\\2out_post_mask.jpg',out)

(x, y, z) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
out = out[topx:bottomx+1, topy:bottomy+1]
cv2.imwrite('Temp\\\\3out_post_colorfill.jpg',out)

###################################
#Find the important Check Numbers:#
###################################
crop_img = out[int(out.shape[0]/(1.2)):out.shape[0], 0:out.shape[1]].copy()
scale_percent = 200 # percent of original size
width = int(crop_img.shape[1] * scale_percent / 100)
height = int(crop_img.shape[0] * scale_percent / 100) 
crop_img_resize = cv2.resize(crop_img, (width, height))
crop_img_gray = cv2.cvtColor(crop_img_resize, cv2.COLOR_BGR2GRAY)
cv2.imwrite('Temp\\\\4cropped_img.jpg',crop_img_gray)
blur = cv2.GaussianBlur(crop_img_gray,(5,5),0)
weightedImg = cv2.addWeighted(crop_img_gray, 1.5, blur, -.5, 1)
cv2.imwrite('Temp\\\\5weightedimg.jpg',weightedImg)
thresholded = cv2.adaptiveThreshold(weightedImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 77, 35)
cv2.imwrite("thresholded.jpg", thresholded)

ret, thresh = cv2.threshold(crop_img_gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
graycopy = crop_img_gray.copy()
secondMax = None
result = []
for i in range(1,len(contours)-2):
        result.append(contours[i])
#print(result)
#cv2.imwrite('Temp\\\\6view_contours_cropped.jpg',crop_img_gray)
new_crop_img_gray = cv2.drawContours(cv2.cvtColor(crop_img_gray, cv2.COLOR_GRAY2BGR), result, -1, (0,255,0), 2)
cv2.imwrite('Temp\\\\6view_contours_cropped.jpg', new_crop_img_gray)

text = tes.image_to_string(thresholded, config='--tessdata-dir \"Training\" -l mcr --oem 1 --psm 3')
transit = text[text.find("a")+1:text.find("a",text.find("a")+1)].replace(" ","")
print("Routing Number: " + transit)
accountNum = text[text.find("a",text.find("a")+1)+1:text.find("c")].replace(" ","")
print("Account Number: " + accountNum)
checkNum = text[text.find("c")+1:len(text)].replace(" ","").replace("\n","")
print("Check Number: " + checkNum)

############################
#Process the number amount:#
############################
crop_img = out[0:out.shape[0], int(out.shape[1]/(1.5)):out.shape[1]].copy()
scale_percent = 200 # percent of original size
width = int(crop_img.shape[1] * scale_percent / 100)
height = int(crop_img.shape[0] * scale_percent / 100) 
crop_img_resize = cv2.resize(crop_img, (width, height))
crop_img_gray = cv2.cvtColor(crop_img_resize, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(crop_img_gray,(5,5),0)
weightedImg = cv2.addWeighted(crop_img_gray, 1.5, blur, -.5, 1)
thresholded = cv2.adaptiveThreshold(weightedImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 77, 35)

ret, thresh = cv2.threshold(crop_img_gray, 175, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
graycopy = crop_img_gray.copy()
secondMax = None
result = []
for i in range(1,len(contours)-2):
        result.append(contours[i])
new_crop_img_gray = cv2.drawContours(cv2.cvtColor(crop_img_gray, cv2.COLOR_GRAY2BGR), result, -1, (0,255,0), 2)
cv2.imwrite('Temp\\\\7view_contours_cropped.jpg', new_crop_img_gray)
max = 0
result = None
for i in range(0,len(contours)):
    if max < cv2.contourArea(contours[i]):
        max = cv2.contourArea(contours[i])
        result = contours[i]
mask = np.zeros_like(new_crop_img_gray)
cv2.drawContours(mask, [result], 0, (255,255,255), -1)
outAmount = np.zeros_like(new_crop_img_gray)
outAmount[mask == 255] = new_crop_img_gray[mask == 255]
cv2.imwrite('Temp\\\\8out_post_mask.jpg',outAmount)

(x, y, z) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
outAmount = outAmount[topx:bottomx+1, topy:bottomy+1]
cv2.imwrite('Temp\\\\9out_post_colorfill.jpg',outAmount)

#Do tesseract things here