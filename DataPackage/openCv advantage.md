# Menu
[[openCv]]
放射变换
warpAffine
warpPectiveTransform

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