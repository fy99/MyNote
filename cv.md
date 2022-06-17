
### Start
```python
cv2.namedWindow()  
cv2.imshow()  
cv2.destroyAllWindows()  
cv2.resizeWindow()  
cv2.waitKey()
```

### Img
```python
img = cv.imread("img/1.jpg")  
cv.imshow("name", img)
cv.imwrite("img/new.jpg", img)  
```

### Video
- read
```python
cap=cv.VideoCapture("img/vi.mp4")  
cap.isOpened()
ret,frame=cap.read()
cap.release()
```

- write
```python
fourcc=cv.VideoWriter_fourcc(*"mp4v")  
vw=cv.VideoWriter("img/new.mp4",fourcc,20,(1280, 720))   # 地址, 格式, 帧率, 分辨率  
vw.write(frame)  
vw.release()
```

### Mouse
- callback
```python
def mouse_callback(event, x, y, flags, userdata):  
    print(event, x, y, flags, userdata)  
cv.setMouseCallback("mouse", mouse_callback, "123")
```

### Trackbar
```python
cv.createTrackbar("R", "windowName", 0, 255, trackbar_callback)
r_val = cv.getTrackbarPos("R", "windowName")
```

### cvtColor
- new_img= cv.cvtColor(img_np, cv.COLOR_BGR2HSL)

### Mat (np.narray)
- dims  [ndim]
- rows [shape[0]]
- cols [shape[1]]
- depth [shape[3]]  [itermsize]
- channels  # rgb = 3
- size [size]
- type dep + dt + chs CV_8UC3 [dtype]
- data [data]

### chanel split & merge
b, g, r = cv.split(img)
cv.merge((b, g, r))

### Draw
1. line (img, pt1, pt2, color, thickness=None, lineType=None, shift=None)
	color: (b, g, r) =(255, 0, 0)
	lineType: 锯齿 (数值越大, 越平滑, 锯齿越少 ) 值应为 2的次方
2. rectangle
3. circle(img, center, radius, color, thickness=None, lineType=None, shift=None)
	center: tupple
	radius: int
	# radius == 0 & thickness/2 为填充圆
1. ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=None, lineType=None, shift=None)
	axes: (x, y)
	angle: 0-360
	startAngle:
	endAngle: 是否画完
5. polylines(img, pts, isClosed, color, thickness=None, lineType=None, shift=None)
	pts: int32 [pts1[ [ int32,int32 ],[int32,int32] ,pts2[ [ int32,int32 ],[int32,int32]] 
	pts1=np.array([(100, 200), (300, 300), (250, 400)],dtype=np.int32)  
	pts2=np.array([(10, 200), (50, 30), (25, 80)],dtype=np.int32)  
	pts=[pts, pts2] => 画两个三角形
6. fillPoly(img, pts, color, lineType=None, shift=None, offset=None)
	1. ~~isClosed~~
	2. ~~thickness~~
7.  putText(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None)
	- fontFace: cv.Font_...
8. 中文	 
	``` python
	from PIL import ImageFont, ImageDraw, Image
	
	#设置字体和大小
	font = ImageFont.truetype('img/msyh.ttc',size=35)  
	
	#将np.array转为pil操作
	img_pil = Image.fromarray(img) 
	
	#选定在img_pil上, 然后写字
	draw =ImageDraw.Draw(img_pil)  
	draw.text((100,150),"你好!",font=font,fill=(0,255,0))
	
	#将img_pil重新转回np.array
	img = np.array(img_pil)
	
	```

  
  
