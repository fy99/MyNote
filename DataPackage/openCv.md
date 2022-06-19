# Part 1

## 1. Start

###  Start
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

## 2. Control
- write
```python
fourcc=cv.VideoWriter_fourcc(*"mp4v")  
vw=cv.VideoWriter("img/new.mp4",fourcc,20,(1280, 720))  # 地址, 格式, 帧率, 分辨率  
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

### Mat narray
- dims  [ndim]
- rows [shape[0]]
- cols [shape[1]]
- depth [shape[3]]  [itermsize]
- channels  # rgb = 3
- size [size]
- type dep + dt + chs CV_8UC3 [dtype]
- data [data]

### Chanel split & merge
b, g, r = cv.split(img)
cv.merge((b, g, r))
## 3. Draw & Change
### Draw
1. line (img, pt1, pt2, color, thickness=None, lineType=None, shift=None)
	color: (b, g, r) =(255, 0, 0)
	lineType: 锯齿 (数值越大, 越平滑, 锯齿越少 ) 值应为 2的次方
2. rectangle
3. circle(img, center, radius, color, thickness=None, lineType=None, shift=None)
	center: tupple
	radius: int
	**radius == 0 & thickness/2 为填充圆**
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

### Caculate
0. mask 
	1. 一个二维数组(没有颜色)
	2. 只计算白色部分
	3. 等同于蒙版
1. resize
	 new_img = cv.resize(img, (x, y))
2. add  直接相加, 当值大于255算过曝, 值改为255 
	cv.add(img1, img2)
3. multiply 
4. subtract  直接相减, 当值小于0时, 值改为255 
5. divide 
6.   addWeighted
	-  new_img=cv.(img1, a,img2, b, 150, gamma)
	-  img1 * a + img2 * b + gamma
7. bit_wise
	- bit_wise_and
	- bit_wise_or
	- bit_wise_xor
	- bit_wise_not
		not: 255 - x

---

### Resize Flip rotate
1. resize
	resize(src. dsize[, dst[, fx[, fyL interpolation]]])
		- src: 要缩放的图片       
		- dsize: 缩放之后的图片大小,元组和列表表示均可，
		- ~~dst: 可选参数,缩放之后的输出图片~~
		- fx,fy:  x轴和y轴的缩放比,即宽度和高度的缩放比.
		- interpolation:插值算法,主要有以下几种:
			INTER_NEAREST,邻近插值,速度快,效果差.
			INTER_LINEAR,双线性插值，使用原图中的4个点进行插值.默认，
			INTER_CUBIC,三次插值,原图中的16个点.
			INTER_AREA,区域插值,效果最好,计算时间最长，

2. flip(src, flip_code)
	- flip_code: 
	
| 操作 | 数值 | np |
| :-: | :-: | :-: |
|上下翻转|0| img[::-1] | 
|左右翻转|>0(1)| img[:,::-1]| 
|上下左右翻转|<0(-1)| img[::-1,::-1]| 

3. rotate
	cv.ROTATE_90_CLOCKWISE => 0
	cv.ROTATE_180 => 1
	cv.ROTATE_90_COUNTERCLOCKWISE =>2
	
### warpAffine

$$
 \begin{bmatrix} x^1 \\ y^1\\1 \end{bmatrix}= 
 \begin{bmatrix} 1 & 0 &t_x \\ 0& 1 & t_y \\0 & 0 & 1\end{bmatrix}
 \begin{bmatrix} x \\ y\\1 \end{bmatrix}
$$

**仿射变换** warp(扭曲) Affine([数学]仿射)
	1. [平移] np.array([(1, 0, 100), (0, 1, 200)], np.float32)
		=========>  x=1x+0y+100   y=0x+1y+200
	2. [旋转] getRotationMatrix2D  逆时针
		M=cv.getRotationMatrix2D(center, angle, scale)
	3. [三角变换] getAffineTransform(src, dst)
		float32 [ [a, b], [c, d], [e, f] ]

```python
warpAffine(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None)
M: float32
                    
M = np.array([(1, 0, 100), (0, 1, 200)], np.float32)

'旋转计算 angle: 逆时针'
M=cv.getRotationMatrix2D(center, angle, scale)


'三角变换 getAffineTransform'
'旧图三个点 && 新图三个点'
src = np.array([[100, 100], [500, 400], [900, 500]], np.float32)  
dst = np.array([[0, 0], [300, 300], [900, 600]], np.float32)  
  
'获取变换矩阵'
M = cv.getAffineTransform(src, dst)  
'进行仿射变换'
new_img = cv.warpAffine(img, M, img.shape[:-1][::-1])  

'将点标记在图上标记'
src_int = np.array(src, "int32")  
dst_int = np.array(dst, "int32")  
img = cv.polylines(img, [src_int], 0, [0, 200, 120], 10, 16)  
new_img = cv.polylines(new_img, [dst_int], 0, [0, 200, 120], 10, 16)

```

### warpPectiveTransform
**透视变换**   **四角变换**
	getPerspectiveTransform
	np.float32([ [左上], [右上], [右下], [左下]], np.float32) 
	**顺序仅需相互对应, 自身次序不重要**




# Part 2
## 1. Filter  & Blur
### Filter2D
1. 卷积 kernel 卷积核
filter2D(src, ddepth, kernel, 
dst=None, anchor=None, delta=None, borderType=None)

- ddepth:
	卷积之后图片的位深,即卷积之后图片的数据类型, 一般设为-1,表示和原图类型一致.
- kernel:
	卷积核大小, 用元组或者ndarray表示,要求数据类型必须是float型.
- anchor 锚点,即卷积核的中心点是可选参数,默认是(-1,-1)
- defta 可选参数,表示卷积之后额外加的一个值,相当于线性方程中的偏差，默认是0.
- borderType边界类型.-般不设.

### BoxFilter
boxFilter(src, ddepth, ksize, dst=None, anchor=None, normalize=None, borderType=None)
-  ksize: kernel 卷积核大小
- normlize=True

### Blur
BoxFilter 的简易操作 (src, ksize)

正太滤波 / 高斯模糊
$G(x,y)=\frac{1}{2 \pi \sigma^2}e^{-\frac{x^2+y^2}{2/\sigma^2}}$

### GaussianBlur
(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
