import cv2
import numpy as np
import pytesseract as tes

#######################################################
#Get a somewhat cleaned image to work with. Base image#
#         to work with will be named "out"            #
#######################################################
check_path_train = './Checks/IMG_3841.JPG'
oimg = cv2.imread(check_path_train, cv2.IMREAD_COLOR)
scale_percent = 50 # percent of original size
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

gray = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,300,400,apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,300)

max_y = 0
for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        if y1 > max_y and y1 < out.shape[0]-10:
                max_y = y1
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

crop_img = out[max_y:out.shape[0], 0:out.shape[1]].copy()
scale_percent = 200 # percent of original size
width = int(crop_img.shape[1] * scale_percent / 100)
height = int(crop_img.shape[0] * scale_percent / 100) 
crop_img_resize = cv2.resize(crop_img, (width, height))
crop_img_gray = cv2.cvtColor(crop_img_resize, cv2.COLOR_BGR2GRAY)
cv2.imwrite('Temp\\\\4cropped_img.jpg',crop_img_gray)
blur = cv2.GaussianBlur(crop_img_gray,(5,5),0)
weightedImg = cv2.addWeighted(crop_img_gray, 1.5, blur, -.5, 1)
cv2.imwrite('Temp\\\\5weightedimg.jpg',weightedImg)
thresholded = cv2.adaptiveThreshold(weightedImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 97, 35)
cv2.imwrite("Temp\\\\6thresholded.jpg", thresholded)

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
thresholded = cv2.adaptiveThreshold(weightedImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 97, 35)

ret, thresh = cv2.threshold(crop_img_gray, 175, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
new_crop_img_gray = cv2.cvtColor(crop_img_gray, cv2.COLOR_GRAY2BGR)
cv2.imwrite('Temp\\\\8view_contours_cropped.jpg', new_crop_img_gray)
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
cv2.imwrite('Temp\\\\9out_post_mask.jpg',outAmount)

(x, y, z) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
outAmount = outAmount[topx:bottomx+1, topy:bottomy+1]
cv2.imwrite('Temp\\\\10out_post_colorfill.jpg',outAmount)

outAmountClean = outAmount[22:outAmount.shape[0]-22, 25:outAmount.shape[1]-25]
scale_percent = 50 # percent of original size
width = int(outAmountClean.shape[1] * scale_percent / 100)
height = int(outAmountClean.shape[0] * scale_percent / 100) 
outAmountClean_resize = cv2.resize(outAmountClean, (width, height))
cv2.imwrite('Temp\\\\11out_post_colorfill.jpg',outAmountClean_resize)

#Do tesseract things here
text = tes.image_to_string(outAmountClean_resize, config='--tessdata-dir \"Training\" -l eng --oem 1 --psm 7')
print("Amount: " + text)

####################
#Do name stuff here#
####################
outName = out.copy()
min_y = 10000
for res in result:
        for x,y in res:
                if y < min_y:
                        min_y = y
outNameCropped = outName[int(min_y/(1.5))-100:int(min_y/(1.55)),225:outName.shape[1]-400]
cv2.imwrite('Temp\\\\12out_Name_Cropped.jpg',outNameCropped)

scale_percent = 50 # percent of original size
width = int(outNameCropped.shape[1] * scale_percent / 100)
height = int(outNameCropped.shape[0] * scale_percent / 100) 
outNameCropped_resize = cv2.resize(outNameCropped, (width, height))
cv2.imwrite('Temp\\\\13out_post_colorfill.jpg',outNameCropped_resize)

text = tes.image_to_string(outNameCropped_resize, config='--tessdata-dir \"Training\" -l eng --oem 1 --psm 12')
print("Name To: " + text)