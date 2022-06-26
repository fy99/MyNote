# Part1

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
#### read
```python
cap=cv.VideoCapture("img/vi.mp4")  
cap.isOpened()
ret,frame=cap.read()
cap.release()
```


#### write
```python
fourcc=cv.VideoWriter_fourcc(*"mp4v")  
vw=cv.VideoWriter("img/new.mp4",fourcc,20,(1280, 720))  # 地址, 格式, 帧率, 分辨率  
vw.write(frame)  
vw.release()
```
## 2. Control
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
0. boxpoint()
#### 1. line 
(img, pt1, pt2, color, thickness=None, lineType=None, shift=None)
	color: (b, g, r) =(255, 0, 0)
	lineType: 锯齿 (数值越大, 越平滑, 锯齿越少 ) 值应为 2的次方
#### 2. rectangle
#### 3. circle
(img, center, radius, color, thickness=None, lineType=None, shift=None)
	center: tupple
	radius: int
	**thickness = -1 为填充圆**
#### 4. ellipse
(img, center, axes, angle, startAngle, endAngle, color, thickness=None, lineType=None, shift=None)
	axes: (x, y)
	angle: 0-360
	startAngle:
	endAngle: 是否画完
#### 5. polylines
(img, pts, isClosed, color, thickness=None, lineType=None, shift=None)
	pts: int32 [pts1[ [ int32,int32 ],[int32,int32] ,pts2[ [ int32,int32 ],[int32,int32]] 
	pts1=np.array([(100, 200), (300, 300), (250, 400)],dtype=np.int32)  
	pts2=np.array([(10, 200), (50, 30), (25, 80)],dtype=np.int32)  
	pts=[pts, pts2] => 画两个三角形
#### 6. fillPoly
(img, pts, color, lineType=None, shift=None, offset=None)
	1. ~~isClosed~~
	2. ~~thickness~~
#### 7.  putText
(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None)
	- fontFace: cv.Font_...
#### 8. 中文	 
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
#### 1. resize
new_img = cv.resize(img, (x, y))
#### 2. add  
直接相加, 当值大于255算过曝, 值改为255 
	cv.add(img1, img2)
3. multiply 
4. subtract  直接相减, 当值小于0时, 值改为255 
5. divide 
#### 6.   addWeighted
- new_img=cv.(img1, a,img2, b, 150, gamma)
- img1 * a + img2 * b + gamma
#### 7. bit_wise
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
	
### warpAffine 仿射变换

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

#### PerspectiveTransform


# Part2 Blur & Sobel
## 1. Filter & Blur
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

### GaussianBlur
正太滤波 / 高斯模糊
$G(x,y)=\frac{1}{2 \pi \sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}$
(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None) kisize: tupple

### MedianBlur
去除椒盐噪点
中位数 : 排序后去中间数的值
medianBlur(src, ksize, dst=None) ksize: int

### BilateralFilter
![[BilateralFilter.png]]
双边滤波: 先查找灰度边界, 然后只在边界相同的一面进行高斯模糊
bilateralFilter(src, d, sigmaColor, sigmaSpace, dst=None, borderType=None)
d: Diameter 直径
sigmaColor: 灰度的边界值 越大边界影响越小
sigmaSpace: 高斯模糊的边界

## 2. sobel 边缘检测算子
$$\begin{bmatrix}-1&0&+1 \\ -2& 0 & +2 \\-1 & 0 & +1\end{bmatrix}$$
边缘: 像素值发生跃迁的位置
Sobel(src, ddepth, dx, dy, dst=None, ksize=None, 
			    scale=None, delta=None, borderType=None)
	ddepth: cv_CV64 / -1
	dx :水平方向 dy: 垂直方向
	ksize: size of the extended Sobel **kernel**

### 2.1 scharr 沙尔
加强版sobel ~~ksize~~  ~~dx=2~~ dx=1
$$\begin{bmatrix}-3&0&+3 \\ -10& 0 & +10 \\-3 & 0 & +3\end{bmatrix}$$

### 2.2 Laplacian 拉普拉斯
更清晰,但抗噪波能力比较差 (img, -1)
$$f''(x,y)=\begin{bmatrix}0&1&0 \\ 1& -4 & 1 \\0 & 1 & 0\end{bmatrix} * k$$
### 2.3 Canny
Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None)
	1. 去噪
	2. 计算梯度 & 8个方向
	4. 相同方向取最大 (非极大值抑制 Non-Maximum Suppression)
	5. 设定阈值,保留高于阈值区, 清除低级阈值区
		  中间区如果和高阈值区相连则保留, 否则移除

## 3. morphology 形态学
### 3.1 Threshold 二值化
trhesh_val, img=threshold(src, thresh, maxval, type, dst=None)
maxval{ int: 255 , float: 1}
| Type                      | Mean                      |
|:------------------------- |:------------------------- |
| cv.THRESH_BINARY          | 标准模式, 大于阈值则为255 |
| cv.THRESH_BINARY_INV      | 标准模式取反              |
| cv.THRESH_TOZERO          | 低于阈值变0  (暗区变0)    |
| cv.THRESH_TOZERO_INV      | TOZERO模式取反 |
| cv.THRESH_TRUNC(truncate) | 高于阈值时变阈值(变暗)    |

###  3.2 AdaptiveThreshold
adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C, dst=None)
adaptiveMethod:
	cv.ADAPTIVE_THRESH_MEAN_C 
	cv.ADAPTIVE_THRESH_GAUSSIAN_C
thresholdType: THRESH_BINARY / THRESH_BINARY_INV
**blockSize: kernel size**
C: 算完之后减去C这个常数

### 3.3 Erode & Dilate 腐蚀&膨胀
rode(src, kernel, dst=None, anchor=None, iterations=None, borderType=None, borderValue=None)
**iterations 重复几次**

Erode: 白色被腐蚀, 黑色膨胀
Dilate: 白色膨胀, 黑色被腐蚀

#### GetStructuringElement 形态卷积核
getStructuringElement(shape, ksize, anchor=None)
kisize:(x,y)
1.  shape == MORPH_RECT 
2.  shape == MORPH_CROSS 
3.  shape == MORPH_ELLIPSE

### 3.4 MorphologyEx
morphologyEx(**src, op, kernel,**
dst=None, anchor=None, **iterations=None**, borderType=None, borderValue=None)
- kernel: GetStructuringElement
- 
#### 3.4.1 OPEN CLOSE 开运算, 闭运算
#### 3.4.2 GRADIENT 形态学梯度
#### 3.4.3 TOPHAT BLACKHAT 顶帽 黑帽

| cv                | 步骤                  | 效果                        |     |
|:----------------- |:--------------------- |:-- |
| cv.erode          | 相邻区有黑色时,变黑色 | 去除白色噪点 白色被腐蚀     |     |
| cv.dilate         | 相邻区有白色时,变白色 | 去除黑色噪点 白色膨胀       |     |
| cv.MORPH_OPEN     | 1. Erode 2. Dilate    | 去除白色噪点                |     |
| cv.MORPH_CLOSE    | 1. Dilate 2. Erode    | 去除黑色噪点                |     |
| cv.MORPH_GRADIENT | img - Erode(img)      | (空心字) 原图减腐蚀后的边界 |     |
| cv.MORPH_TOPHAT   | img - OPEN(img)       | 得到白色噪点                |     |
| cv.MORPH_BLACKHAT | img - CLOSE(img)      | 得到黑色噪点                |     |

## 4. Contours 边缘
### 4.1 findContours
(image, mode, method, contours=None, hierarchy=None, offset=None)

新版本不返回result_img,
contours -> list[ ndarray [ [ [ int32, int32 ], ] ] shape(n,1,2),  
hierarchy 层级 -> ndarray[ [ [ int32, int32, int32, int32 ], ] ] shape(1,n,4)

| Mode           | 数值 | 效果                                    |
|:-------------- |:---- |:--------------------------------------- |
| RETR_CCOMP     | 2    | 2每层最多两级,从小到大,从里到外(不常用) |
| RETR_EXTERNAL  | 0    | 0值检测外围轮廓                         |
| RETR_FLOODFILL | 4    |                                         |
| RETR_LIST      | 1    | 1检测所有轮廓,但不建立等级关系          |
| RETR_TREE      | 3    | 树形,从大到小,从右到左                  |

| method              | 数值 | 效果                 |
|:------------------- |:---- |:-------------------- |
| CHAIN_APPROX_NONE   | 1    | 保存全部边缘线上的点 |
| CHAIN_APPROX_SIMPLE | 2    | 只保存角上的点       |

### 4.2 drawContours 
drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
contourIdx: -1表示画出所有

### 4.3 contourArea 选定区域面积
计算单个contour的面积, 返回单位像素的数值
contourArea(contour)

### 4.4 arcLength 周长
计算单个contour的周长
cv.arcLength(contour,closed=Faslse)

### 4.5 approxPolyDP 多边形逼近
approxPolyDP(curve, epsilon, closed, approxCurve=None)
curve: contour
epsilon: 点与所画直线的最远阈值

### 4.6 convexHull 凸包
(contour=nddarry)

### 4.7 minAeraRect 自小矩形
boxFR: [ [ x:float, y , w, h ],rotate:float ]=(contour=nddarry)

#### Draw: boxPoint
4pointF = botPoint(box)
polylines=np.round(4pointF).astype("int64")
cv.polylines()

### 4.8 boundingRect 矩形轮廓
x, y, w, h = boundingRect(contour)
cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 240), 4)

### 4.9. pyramid 金字塔
#### Gaussian pyramid
 pyrdown() 1/4
 pyrup() 
#### Laplacian pyramid
img_orgin - pyrup( pyrdown( img ) )
### 4.10 calcHist 直方图
calcHist(images, channels, mask, histSize, ranges, hist=None, 
accumulate=None)
mask: 蒙版
images: [ ],
channels: [ 通道 ]
histSize: [ bins ]
range:[ ]
accumulate: 累计, 不清零上次统计数据
### 4.11 equalizeHist 自动色阶
img: 只能是单一通道

### 4.12 BackgroundSubtractorMOG()
cv.bgsegm.create**BackgroundSubtractorMOG**

# P3 Match
## 1. Cornel
### 1.1 cornerHarris
(src, blockSize, ksize, k, dst=None, borderType=None)->ndarry

blockSize: 扫描区域大小, 可以是整数 类似卷积核
ksize: sobel算子大小, 用于检查灰度的梯度
k:  a b- k( a + b ) ** 2 一般取0.04, 范围在 0.04~0.06
return: src相同shape的ndarry
#### [corners > corners.max() * 0.02]
取出比 (最大角点*0.2) 大的角点的 bool ndarry

### 1.2 shi-Tomals=>goodFeaturesToTrack
(image, maxCorners, qualityLevel, minDistance, corners=None, mask=None, blockSize=None, useHarrisDetector=None, k=None)

maxCorners: 最多多少个
qualityLevel: 角的质量,0~1之间, 一般为 0.01~0.1
minDistance: 两个角之间的最小欧式距离 $\sqrt{\sum{ (x_1^2+y_1^2)}}$

不用设置 K R=min(a, b)
可以直接设置阈值 qualityLevel
return: list[ list[ list_pos[float: x, float: y ] ] ]
## 2. Feature  关键点和描述子
1. keypoints, = sift.detect(gray_img)  
2. keypoints, descriptions = sift.compute(gray_img, kp)	
3. **keypoints, descriptions = sift.detectAndCompute(gray_img, mask=None)**
4. cv.drawKeypoints(gray_img, kp, img)

keypoint:
	1. angle : float
	2. class_id : int
	3. octave : int
	4. pt :(float, float)
	5. response: float
	6. size: float

descriptions:
shape: (n, 128) ->[ [float32 ] * 128 , ]
	以关键点为中心, 取4x4个区域, 每个区域单独计算8个种子点
	4 x 4 x 8
| name | feature     | speed      | num       | time   |
| ---- | ----------- | ---------- | --------- |:------ |
| SIFT | 均衡        | 一般       | 1         | 1      |
| SURF | 特征多      | (理论上快) | 1.2-0.95  | 1.87   |
| ORB  | 特征少,更快 | 极快       | 0.15-0.07 | 0.0686 |




### 2.1 SIFT=>Scale-invariant feature transform
多层金字塔, 多个高斯模糊图片
sift=cv2.xfeatures2d.SIFT_create()  
sift.detectAndCompute(img,mask=None)

### 2.2 SURF=>Speeded-Up Robust Features
surf = cv.xfeatures2d.SURF_create()
SIFT 准确 慢
SURF 更快

### 2.3 ORB=>Oriented FAST and Rotated BRIEF

1 可以做到实时检测 ( 更快 ) 2. 开源 
FAST :实时检测 -> 
	Oriented 加入方向
BRIEF:对已检测的特征点进行描述 加快了特征符的建立速度->
	Rotated  加入了方向性

## 3. Match
1. ...Matcher()
2. bf.match(des1,des2)
	1. des1 一般是小图
	dmatch= 	list(	dict (
		distance = float, 
		imgIdx = int
		queryIdx = int , 
		trainIdx = int) )

3. cv.drawMatches(img1, kp1, img2, kp2 ,match_rst)
	1. img1, kp1 搜索的图 
	2.  img2, kp2 匹配的图
	3. match_rst 返回的结果



### 3.1 BF=>Brute Force 暴力匹配
枚举 慢
bf=BFMatcher(normType, crossCheck)
	normType:
		NORM_L1, 绝对值相加
		NORM_L2(默认),  => SIFT/SURF 平方和在开方 欧式距离
	    HAMMING1, 位判断
	    HAMMING2,
	crossCheck:
		是否交叉(互相2次)匹配,  默认False

### 3.2 FLANN
快 精度差
index_parmas = {  
    "alogrithm": 1,  
    "trees": 5,  
}  
search_parmas = {  "checks": 50 }  
cv.FlannBasedMatcher(index_parmas, search_parmas)

index_params:
	KDTREE=> sift surf
		index_params=dict{
			algorithm=FLANN_INDEX_KDTREE,
			trees=5
		}
		search_params=>遍历次数 {checks:50}
			KDTREE:5 search_times:10
	LSH => orb
		index_params=dict{
			algorithm=LSH	
		}

match(des1,des2) => cv.drawMatches
KnnMatch() => cv.drawMatchesKnn

KnnMatch(des1, des2, k) -> DMatch
	K:第一幅图的kp与匹配的第二幅图的描述子
			欧式距离最近(最匹配的)前K个关键点kp
	DMatch:
		distance: des之间的距离, 越近越好
		queryIdx: 第一个图像的descriptions索引值
		trainIdx: 第二个图的des索引值
		imgIdx: 第二个图的索引值
drawMatchesKnn(img1, kp1, img2, kp2, DMatch)
```python
'KnnMatch'
index_parmas = {  
    "alogrithm": 1,  
    "trees": 5,  
}  
search_parmas = {"checks": 50}  
cv.FlannBasedMatcher(index_parmas, search_parmas)

good_dmatch = []  
for x, y in dmatch:  
    if x.distance < y.distance * 0.7:  
        good_dmatch.append(x)  

img4 = cv.drawMatchesKnn(img_matcher, kp1, img, kp, dmatch, None)
img4 = cv.drawMatchesKnn(img_matcher, kp1, img, kp, [good_dmatch], None)
```

### 3.2 Match
```python
bf = cv.BFMatcher()  
rst = bf.match(des1, des2)  
  
if len(rst) >= 4:  
    src_pts = np.float32([kp1[i.queryIdx].pt for i in rst]).reshape(-1, 1, 2)  
    dst_pts = np.float32([kp2[i.trainIdx].pt for i in rst]).reshape(-1, 1, 2)  
  
    h, w = img1.shape[:2]  
     # 随机抽样抑制算法  
    H, _mask = cv.findHomography(src_pts, dst_pts, 
	    method=cv.RANSAC, ransacReprojThreshold=5) 
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
	     [w - 1, 0]]).reshape(-1, 1, 2)   
  
    dst = cv.perspectiveTransform(pts, H)  
    cv.polylines(img2, [np.int32(dst)], True, (0, 0, 255), 3)  
  
img3 = cv.drawMatches(img1, kp1, img2, kp2, rst, None)  
cv.imshow("img3", img3)
```

### == 图像拼接
```python
'图像拼接'
import cv2  
import cv2 as cv  
import numpy as np  
  
img1 = cv.imread('img/l.jpg')  
img2 = cv.imread('img/r.jpg')  
# img1 = img2[400:600, 400:600]  
  
  
img_gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)  
img_gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)  
  
sift = cv.SIFT_create()  
kp1, des1 = sift.detectAndCompute(img_gray1, None)  
kp2, des2 = sift.detectAndCompute(img_gray2, None)  
  
bf = cv.BFMatcher()  
dMatchs = bf.match(des1, des2)  
  
  
def stitch_img(img1, img2, H):  
    h1, w1 = img1.shape[:-1]  
    h2, w2 = img2.shape[:-1]  
  
    img1_dims = np.float_([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)  
    img2_dims = np.float_([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)  
  
    img1_transform = cv.perspectiveTransform(img1_dims, H)  
    img2_transform = cv.perspectiveTransform(img1_dims, H)  
    print(f"{img1_transform=}")  
    # print(f"{img2_transform=}")  
  
    # 第二张图的原始位置和第一张图的变化后的位置  
    result_dims = np.concatenate((img2_dims, img1_transform), axis=0)  
    print(f"{result_dims=}")  
  
    x_max, y_max = np.round(result_dims.max(0).ravel(), ).astype("i8")  
    x_min, y_min = np.round(result_dims.min(0).ravel(), ).astype("i8")  
  
    # transform_dist = [-x_min, -y_min]  
    transform_array = np.float_([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])  
  
    result_img = cv.warpPerspective(img1, transform_array.dot(H), (x_max - x_min, y_max - y_min))  
  
    result_img[-y_min:-y_min + h2, -x_min:-x_min + w2] = img2  
    return result_img  
  
  
print(f"{dMatchs=}")  
if len(dMatchs) >= 5:  
    src_pts = np.float32([kp1[i.queryIdx].pt for i in dMatchs]).reshape(-1, 1, 2)  
    dst_pts = np.float32([kp2[i.trainIdx].pt for i in dMatchs]).reshape(-1, 1, 2)  
  
    print(src_pts)  
    print(dst_pts)  
    print(f"{len(src_pts)=}")  
    print(f"{len(dst_pts)=}")  
  
    h, w = img1.shape[:2]  
  
    # 随机抽样抑制算法  
    H, _mask = cv.findHomography(src_pts, dst_pts, method=cv.RANSAC, ransacReprojThreshold=5)  
  
    img_tran = stitch_img(img1, img2, H)  
    cv.imshow("connected_img", img_tran)  
    cv.waitKey()  
    cv.destroyAllWindows()  
  
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)  
    dst = cv.perspectiveTransform(pts, H)  
  
    print(f"{dst=}")  
  
    cv.polylines(img_gray2, [np.int32(dst)], True, (0, 0, 255), 3)  
  
img3 = cv.drawMatches(img_gray1, kp1, img_gray2, kp2, dMatchs, None)  
cv.imshow("img3", img3)  
cv.waitKey()  
cv.destroyAllWindows()
```