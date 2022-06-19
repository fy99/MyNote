#  Numpy
## important
- .view() 浅拷贝 可以以不同类型显示(数据相同)
- .copy() 深拷贝

- np.stack() 堆叠 
	- 不改变维度
		1. row_stack == vstack 竖着堆
		2. colom_stack == hstack 横着堆
	
	- 增加一维度
		1. stack  直接堆叠
		2. dstack 拆开每个元素 然后堆叠

	- 降低一个维度
		np.concatenate([ [1,2], [3,4] ], axis) =>[1, 2, 3, 4] 


### First
ndarray
- .ndim 维度 dimension       ndmin = 2  设置为二维
- .shape (x,y)       
- .size       
- .dtype       
- .itemsize


```python
ndarray.ndim = 2  
ndarray.shape = (10, 10)  
ndarray.size  100  
ndarray.dtype = dtype('float64')  
ndarray.itemsize = 8字节 int32 = "i4" float64 = "f8" int8 ="i1"(127, -128)
```

### make
- np.array()
- np.arange()
- np.linspace() 等差数列
- np.logspace() 等比数列
- np.zeros() 
- np.ones()
- np.zeros_like()
- np.r_[]
- np.c_[]
- np.s_[]
- np.copy()

### select


```PYTHON
[1][2] 取一个数  
*  [...] 等于没有写  
[...,:]  
[...,::-1] 


[[1,2,3],[2,3,4]] 相当于分次  
[[1,2,3],[True,False,...,True]] 相当于分次  
3行4列  
a[[True,False,True],:][:,[True,False,True,False]  
a[[0,2],:][:,[0,2]]  
  
  
[*>6] a[a>6]  
a[a%2==0]  找出所有偶数 并显示  
a[a==5]=1  把a中所有5替换为1  
b = a%2==1 bool数组赋值到b中  
a[(a>5)&(a<10]  
a[(a>5)|(a<3]  
a[~(a>5)]  
a[:,3] != a[:,[3]] 横着一行 和 多行每行只有一列  
```

### 广播机制  

```
运算时小数组(仅限一维[无论行或列]的数组)会复制成相同大小  
z,y,x (块数,行数,列数)  
=====>右对齐  
<<<右对齐>>>  
两个比较  
相同维度中  
1:相等  
2:有一个1 (结果的shape取最大数)  
3:有一个为空 (空的本质就是1)  
  
  
, 约等于 ][a[...][0]=a[0]  
a[..., 0]  
: == ...  
[[0,2,4],[1,3,5]]=(0,1),(2,3),(4,5)  
```

## 计算
### 平均
n.mean() average  
axis 0 竖着列的平均 1 行的平均  
右对齐 从0数  
  
### 中位数  
np.median(a)  
  
### 标准差 (方差 np.var 的根)  
stand different StD 
np.std()  

### 其他计算
max min sum cumsum(累加求和)  
a.average 加权平均数 weights


## Dtype
### 数据类型  
  
bool            == b  
int8/16...64    == i1 i2 i4  
float8/16...64  == f1 f2 f4  
uint8 无符号整数  == u1 u2 u4str   字符串      == a string  
=>byte 字节串  类型   == S  
complex 复数     == c  
timedelta   == m  
datatime    == M  
  
object      == o  
  
str         == a  
String      == S  
unicode     == U  
  
void        == V


## np.random
### 0-1 随机  
- n = np.random.rand(1000)  
  
### 正太分布 normal distribution
- n = np.random.normal(size=1000, loc=50, scale=0.2)  

### 随机整数  
- n=np.random.randint(0,101,(2,1000),"u1")  
  
### 设置种子  
- np.random.seed(1)


## Change
- resize 
- append 
- insert  
### del
- np.delete(n,5) 返回一个一维数组  
- np.delete(n,[3,4,5]) 返回一个一维数组  
- np.delete(n,list|int,axis) 删除某一列 (axis = 0) 或 某一行 ( axis = 1)np.c_[] 转为/生成列 
-  

### 查找位置  
np.argwhere()  
np.where()  
  
### 去重  
np.unique  

### 截取小数
- np.around()    
- np.trunc()
- np.fix()


### F
arg 位置有关  
np.r_[] 转为/生成行  
np.c_[]
np.s_ []  切片  
sort