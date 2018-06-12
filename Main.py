import cv2
import os
import matplotlib
from matplotlib import pyplot as plt
import numpy

# original(3), edge(2), crop(2), thres(2)
font = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **font)

def namestr(obj, namespace):
    listOfVariables = [name for name in namespace if namespace[name] is obj]
    return listOfVariables[0]

def ReadAndResize(image):
    img = cv2.imread(image, 0)
    imgRes = img.shape
    return cv2.resize(img, (int(imgRes[1] / 5), int(imgRes[0] / 5)))


def TempMatch(src, temp):
    result = cv2.matchTemplate(src, temp, cv2.TM_CCOEFF)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    w, h = temp.shape[::-1]
    topLeft = maxLoc
    bottomRight = (topLeft[0] + w, topLeft[1] + h)
    return (topLeft, bottomRight)


def autoCanny(image, sigma=0.33):
    v = numpy.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def ShowImages(subIndex, image, title, _cmap, showTicks=False):
	plt.subplot(subIndex)
	plt.imshow(image, cmap=_cmap)
	plt.title(title)
	if(not showTicks):
		plt.xticks([]), plt.yticks([])

taperCopy = cv2.imread(os.path.join('Pics', 'Tapered.jpg'))
okCopy = cv2.imread(os.path.join('Pics', 'Ok.jpg'))

taper = cv2.imread(os.path.join('Pics', 'Tapered.jpg'), 0)
ok = cv2.imread(os.path.join('Pics', 'Ok.jpg'), 0)

template = cv2.imread(os.path.join('Pics', 'Ok_template.jpg'), 0)

okMatchLoc = TempMatch(ok, template)
taperMatchLoc = TempMatch(taper, template)

# cv2.rectangle(ok, okMatchLoc[0], okMatchLoc[1], (0,255,0), 5)
# cv2.rectangle(taper, taperMatchLoc[0], taperMatchLoc[1], (0,255,0), 5)

ok = ok[okMatchLoc[0][1]:okMatchLoc[1][1], okMatchLoc[0][0]:okMatchLoc[1][0]]
taper = taper[taperMatchLoc[0][1]:taperMatchLoc[1][1], taperMatchLoc[0][0]:taperMatchLoc[1][0]]

k = 49
kernel = (k, k)

ok = cv2.GaussianBlur(ok, kernel, 0)
ok = cv2.bitwise_not(ok)
ret, ok = cv2.threshold(ok, 150, 255, cv2.THRESH_BINARY)
ok = autoCanny(ok)


taper = cv2.GaussianBlur(taper, kernel, 0)
taper = cv2.bitwise_not(taper)
ret, taper = cv2.threshold(taper, 150, 255, cv2.THRESH_BINARY)
taper = autoCanny(taper)

# Cropping to the edge
okEdge = ok[300:350, 100:300]
taperEdge = taper[300:350, 100:300]

# box = 0 0 -> 200 50
boundary = 25
okCheckBox = okEdge[0:boundary, 0:200]
taperCheckBox = taperEdge[0:boundary, 0:200]

# Checking matrix sum
print('Ok Sum: {} \nTaper Sum: {}\n'.format(
    okCheckBox.sum(), taperCheckBox.sum()))

# Dictionary with reference to all images to show and their descriptions
blank = numpy.zeros((100,100,3))
imageList = [template, ok, taper, okEdge, taperEdge, okCheckBox, taperCheckBox,okCopy, taperCopy]

imageDict = {
	'okCopy': 'Ok image',
	'taperCopy' : 'Tapered coil Image',
	'template' : 'Matching Template',
	'ok' : 'Ok image edged',
	'taper' : 'Taper image edged',
	'blank' : 'Blank',
	'okEdge' : 'Ok edged image zoomed',
	'taperEdge' : 'Taper edged image zoomed',
	'okCheckBox' : 'Ok Box',
	'taperCheckBox' : 'Taper Box',
}


# Finally determining OK or NG
font = cv2.FONT_HERSHEY_DUPLEX
check = 1000
textPos = (300, 200)
samples = [[okCopy,okCheckBox], [taperCopy,taperCheckBox]]
for sample in samples:
	if sample[1].sum() > check:
		cv2.putText(sample[0], 'NG', textPos, font, 2, (255,0,0), 2,cv2.LINE_AA)
	else:
		cv2.putText(sample[0], 'OK', textPos, font, 2, (0,255,0), 2,cv2.LINE_AA)


# Plotting images
subPlotIndex = 251
i=0
for image in imageList:
	ShowImages(subPlotIndex + i,image, imageDict[namestr(image,globals())], 'gray', True)
	i+=1

plt.show()

f = open('variables.txt','w')
f.write(str(globals()))
f.close()

cv2.waitKey(0)
cv2.destroyAllWindows()