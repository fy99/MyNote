### 1
- [x] 1、cvLoadImage：将图像文件加载至内存；  

- [x] 2、cvNamedWindow：在屏幕上创建一个窗口；

- [x] 3、cvShowImage：在一个已创建好的窗口中显示图像；

- [x] 4、cvWaitKey：使程序暂停，等待用户触发一个按键操作；

- [x] 5、cvReleaseImage：释放图像文件所分配的内存；

- [x] 6、cvDestroyWindow：销毁显示图像文件的窗口；

- [x] 7、cvCreateFileCapture：通过参数设置确定要读入的AVI文件；

- [x] 8、cvQueryFrame：用来将下一帧视频文件载入内存；

- [x] 9、cvReleaseCapture：释放CvCapture结构开辟的内存空间；

- [x] 10、cvCreateTrackbar：创建一个滚动条；

- [ ] 11、cvSetCaptureProperty：设置CvCapture对象的各种属性；

- [ ] 12、cvGetCaptureProperty：查询CvCapture对象的各种属性；

- [x] 13、cvGetSize：当前图像结构的大小；

- [ ] 14、cvSmooth：对图像进行平滑处理；

- [ ] 15、cvPyrDown：图像金字塔，降采样，图像缩小为原来四分之一；

- [ ] 16、cvCanny：Canny边缘检测；

- [ ] 17、cvCreateCameraCapture：从摄像设备中读入数据；

- [ ] 18、cvCreateVideoWriter：创建一个写入设备以便逐帧将视频流写入视频文件；

- [ ] 19、cvWriteFrame：逐帧将视频流写入文件；

- [ ] 20、cvReleaseVideoWriter：释放CvVideoWriter结构开辟的内存空间；

- [ ] 21、CV_MAT_ELEM：从矩阵中得到一个元素；

- [ ] 22、cvAbs：计算数组中所有元素的绝对值；

- [ ] 23、cvAbsDiff：计算两个数组差值的绝对值；

- [ ] 24、cvAbsDiffS：计算数组和标量差值的绝对值；

- [ ] 25、cvAdd：两个数组的元素级的加运算；

- [ ] 26、cvAddS：一个数组和一个标量的元素级的相加运算；

- [ ] 27、cvAddWeighted：两个数组的元素级的加权相加运算(alpha运算)；

- [ ] 28、cvAvg：计算数组中所有元素的平均值；

- [ ] 29、cvAvgSdv：计算数组中所有元素的绝对值和标准差；

- [ ] 30、cvCalcCovarMatrix：计算一组n维空间向量的协方差；

- [ ] 31、cvCmp：对两个数组中的所有元素运用设置的比较操作；

- [ ] 32、cvCmpS：对数组和标量运用设置的比较操作；

- [ ] 33、cvConvertScale：用可选的缩放值转换数组元素类型；

- [ ] 34、cvCopy：把数组中的值复制到另一个数组中；

- [ ] 35、cvCountNonZero：计算数组中非0值的个数；

- [ ] 36、cvCrossProduct：计算两个三维向量的向量积(叉积)；

- [ ] 37、cvCvtColor：将数组的通道从一个颜色空间转换另外一个颜色空间；

- [ ] 38、cvDet：计算方阵的行列式；

- [ ] 39、cvDiv：用另外一个数组对一个数组进行元素级的除法运算；

- [ ] 40、cvDotProduct：计算两个向量的点积；

- [ ] 41、cvEigenVV：计算方阵的特征值和特征向量；

- [ ] 42、cvFlip：围绕选定轴翻转；

- [ ] 43、cvGEMM：矩阵乘法；

- [ ] 44、cvGetCol：从一个数组的列中复制元素；

- [ ] 45、cvGetCols：从数据的相邻的多列中复制元素；

- [ ] 46、cvGetDiag：复制数组中对角线上的所有元素；

- [ ] 47、cvGetDims：返回数组的维数；

- [ ] 48、cvGetDimSize：返回一个数组的所有维的大小；

- [ ] 49、cvGetRow：从一个数组的行中复制元素值；

- [ ] 50、cvGetRows：从一个数组的多个相邻的行中复制元素值；

- [ ] ![](https://nimg.ws.126.net/?url=http%3A%2F%2Fdingyue.ws.126.net%2F2021%2F0508%2Fa740e830p00qsqvj900g4d200u000btg00id0078.png&thumbnail=660x2147483647&quality=80&type=jpg)  

### 50
- [ ] 51、cvGetSize：得到二维的数组的尺寸，以CvSize返回；

- [ ] 52、cvGetSubRect：从一个数组的子区域复制元素值；

- [ ] 53、cvInRange：检查一个数组的元素是否在另外两个数组中的值的范围内；

- [ ] 54、cvInRangeS：检查一个数组的元素的值是否在另外两个标量的范围内；

- [ ] 55、cvInvert：求矩阵的逆；

- [ ] 56、cvMahalonobis：计算两个向量间的马氏距离；

- [ ] 57、cvMax：在两个数组中进行元素级的取最大值操作；

- [ ] 58、cvMaxS：在一个数组和一个标量中进行元素级的取最大值操作；

- [ ] 59、cvMerge：把几个单通道图像合并为一个多通道图像；

- [ ] 60、cvMin：在两个数组中进行元素级的取最小值操作；

- [ ] 61、cvMinS：在一个数组和一个标量中进行元素级的取最小值操作；

- [ ] 62、cvMinMaxLoc：寻找数组中的最大最小值；

- [ ] 63、cvMul：计算两个数组的元素级的乘积(点乘)；

- [ ] 64、cvNot：按位对数组中的每一个元素求反；

- [ ] 65、cvNormalize：将数组中元素进行归一化；

- [ ] 66、cvOr：对两个数组进行按位或操作；

- [ ] 67、cvOrs：在数组与标量之间进行按位或操作；

- [ ] 68、cvReduce：通过给定的操作符将二维数组简为向量；

- [ ] 69、cvRepeat：以平铺的方式进行数组复制；

- [ ] 70、cvSet：用给定值初始化数组；

- [ ] 71、cvSetZero：将数组中所有元素初始化为0；

- [ ] 72、cvSetIdentity：将数组中对角线上的元素设为1，其他置0；

- [ ] 73、cvSolve：求出线性方程组的解；

- [ ] 74、cvSplit：将多通道数组分割成多个单通道数组；

- [ ] 75、cvSub：两个数组元素级的相减；

- [ ] 76、cvSubS：元素级的从数组中减去标量；

- [ ] 77、cvSubRS：元素级的从标量中减去数组；

- [ ] 78、cvSum：对数组中的所有元素求和；

- [ ] 79、cvSVD：二维矩阵的奇异值分解；

- [ ] 80、cvSVBkSb：奇异值回代计算；

- [ ] 81、cvTrace：计算矩阵迹；

- [ ] 82、cvTranspose：矩阵的转置运算；

- [ ] 83、cvXor：对两个数组进行按位异或操作；

- [ ] 84、cvXorS：在数组和标量之间进行按位异或操作；

- [ ] 85、cvZero：将所有数组中的元素置为0；

- [ ] 86、cvConvertScaleAbs：计算可选的缩放值的绝对值之后再转换数组元素的类型；

- [ ] 87、cvNorm：计算数组的绝对范数， 绝对差分范数或者相对差分范数；

- [ ] 88、cvAnd：对两个数组进行按位与操作；

- [ ] 89、cvAndS：在数组和标量之间进行按位与操作；

- [ ] 90、cvScale：是cvConvertScale的一个宏，可以用来重新调整数组的内容，并且可以将参数从一种数据类型转换为另一种；

- [ ] 91、cvT：是函数cvTranspose的缩写；

- [ ] 92、cvLine：画直线；

- [ ] 93、cvRectangle：画矩形；

- [ ] 94、cvCircle：画圆；

- [ ] 95、cvEllipse：画椭圆；

- [ ] 96、cvEllipseBox：使用外接矩形描述椭圆；

- [ ] 97、cvFillPoly、cvFillConvexPoly、cvPolyLine：画多边形；

- [ ] 98、cvPutText：在图像上输出一些文本；

- [ ] 99、cvInitFont：采用一组参数配置一些用于屏幕输出的基本个特定字体；

- [ ] 100、cvSave：矩阵保存；

- [ ] ![](https://nimg.ws.126.net/?url=http%3A%2F%2Fdingyue.ws.126.net%2F2021%2F0508%2F820f69a9p00qsqvja006yd200fo006lg00id007p.png&thumbnail=660x2147483647&quality=80&type=jpg)  
### 100
- [ ] 101、cvLoad：矩阵读取；

- [ ] 102、cvOpenFileStorage：为读/写打开存储文件；

- [ ] 103、cvReleaseFileStorage：释放存储的数据；

- [ ] 104、cvStartWriteStruct：开始写入新的数据结构；

- [ ] 105、cvEndWriteStruct：结束写入数据结构；

- [ ] 106、cvWriteInt：写入整数型；

- [ ] 107、cvWriteReal：写入浮点型；

- [ ] 108、cvWriteString：写入字符型；

- [ ] 109、cvWriteComment：写一个XML或YAML的注释字串；

- [ ] 110、cvWrite：写一个对象；

- [ ] 111、cvWriteRawData：写入多个数值；

- [ ] 112、cvWriteFileNode：将文件节点写入另一个文件存储器；

- [ ] 113、cvGetRootFileNode：获取存储器最顶层的节点；

- [ ] 114、cvGetFileNodeByName：在映图或存储器中找到相应节点；

- [ ] 115、cvGetHashedKey：为名称返回一个惟一的指针；

- [ ] 116、cvGetFileNode：在映图或文件存储器中找到节点；

- [ ] 117、cvGetFileNodeName：返回文件的节点名；

- [ ] 118、cvReadInt：读取一个无名称的整数型；

- [ ] 119、cvReadIntByName：读取一个有名称的整数型；

- [ ] 120、cvReadReal：读取一个无名称的浮点型；

- [ ] 121、cvReadRealByName：读取一个有名称的浮点型；

- [ ] 122、cvReadString：从文件节点中寻找字符串；

- [ ] 123、cvReadStringByName：找到一个有名称的文件节点并返回它；

- [ ] 124、cvRead：将对象解码并返回它的指针；

- [ ] 125、cvReadByName：找到对象并解码；

- [ ] 126、cvReadRawData：读取多个数值；

- [ ] 127、cvStartReadRawData：初始化文件节点序列的读取；

- [ ] 128、cvReadRawDataSlice：读取文件节点的内容；

- [ ] 129、cvGetModuleInfo：检查IPP库是否已经正常安装并且检验运行是否正常；

- [ ] 130、cvResizeWindow：用来调整窗口的大小；

- [ ] 131、cvSaveImage：保存图像；

- [ ] 132、cvMoveWindow：将窗口移动到其左上角为x,y的位置；

- [ ] 133、cvDestroyAllWindow：用来关闭所有窗口并释放窗口相关的内存空间；

- [ ] 134、cvGetTrackbarPos：读取滑动条的值；

- [ ] 135、cvSetTrackbarPos：设置滑动条的值；

- [ ] 136、cvGrabFrame：用于快速将视频帧读入内存；

- [ ] 137、cvRetrieveFrame：对读入帧做所有必须的处理；

- [ ] 138、cvConvertImage：用于在常用的不同图像格式之间转换；

- [ ] 139、cvErode：形态腐蚀；

- [ ] 140、cvDilate：形态学膨胀；

- [ ] 141、cvMorphologyEx：更通用的形态学函数；

- [ ] 142、cvFloodFill：漫水填充算法，用来进一步控制哪些区域将被填充颜色；

- [ ] 143、cvResize：放大或缩小图像；

- [ ] 144、cvPyrUp：图像金字塔，将现有的图像在每个维度上都放大两倍；

- [ ] 145、cvPyrSegmentation：利用金字塔实现图像分割；

- [ ] 146、cvThreshold：图像阈值化；

- [ ] 147、cvAcc：可以将8位整数类型图像累加为浮点图像；

- [ ] 148、cvAdaptiveThreshold：图像自适应阈值；

- [ ] 149、cvFilter2D：图像卷积；

- [ ] 150、cvCopyMakeBorder：将特定的图像轻微变大，然后以各种方式自动填充图像边界；

- [ ] ![](https://nimg.ws.126.net/?url=http%3A%2F%2Fdingyue.ws.126.net%2F2021%2F0508%2Fcebe7157p00qsqvja0032d200l800egg00id00ch.png&thumbnail=660x2147483647&quality=80&type=jpg)  
### 150
- [ ] 151、cvSobel：图像边缘检测，Sobel算子；

- [ ] 152、cvLaplace：拉普拉斯变换、图像边缘检测；

- [ ] 153、cvHoughLines2：霍夫直线变换；

- [ ] 154、cvHoughCircles：霍夫圆变换；

- [ ] 155、cvRemap：图像重映射，校正标定图像，图像插值；

- [ ] 156、cvWarpAffine：稠密仿射变换；

- [ ] 157、cvGetQuadrangleSubPix：仿射变换；

- [ ] 158、cvGetAffineTransform：仿射映射矩阵的计算；

- [ ] 159、cvCloneImage：将整个IplImage结构复制到新的IplImage中；

- [ ] 160、cv2DRotationMatrix：仿射映射矩阵的计算；

- [ ] 161、cvTransform：稀疏仿射变换；

- [ ] 162、cvWarpPerspective：密集透视变换(单应性)；

- [ ] 163、cvGetPerspectiveTransform：计算透视映射矩阵；

- [ ] 164、cvPerspectiveTransform：稀疏透视变换；

- [ ] 165、cvCartToPolar：将数值从笛卡尔空间到极坐标(极性空间)进行映射；

- [ ] 166、cvPolarToCart：将数值从极性空间到笛卡尔空间进行映射；

- [ ] 167、cvLogPolar：对数极坐标变换；

- [ ] 168、cvDFT：离散傅里叶变换；

- [ ] 169、cvMulSpectrums：频谱乘法；

- [ ] 170、cvDCT：离散余弦变换；

- [ ] 171、cvIntegral：计算积分图像；

- [ ] 172、cvDistTransform：图像的距离变换；

- [ ] 173、cvEqualizeHist：直方图均衡化；

- [ ] 174、cvCreateHist：创建一新直方图；

- [ ] 175、cvMakeHistHeaderForArray：根据已给出的数据创建直方图；

- [ ] 176、cvNormalizeHist：归一化直方图；

- [ ] 177、cvThreshHist：直方图阈值函数；

- [ ] 178、cvCalcHist：从图像中自动计算直方图；

- [ ] 179、cvCompareHist：用于对比两个直方图的相似度；

- [ ] 180、cvCalcEMD2：陆地移动距离(EMD)算法；

- [ ] 181、cvCalcBackProject：反向投影；

- [ ] 182、cvCalcBackProjectPatch：图块的方向投影；

- [ ] 183、cvMatchTemplate：模板匹配；

- [ ] 184、cvCreateMemStorage：用于创建一个内存存储器；

- [ ] 185、cvCreateSeq：创建序列；

- [ ] 186、cvSeqInvert：将序列进行逆序操作；

- [ ] 187、cvCvtSeqToArray：复制序列的全部或部分到一个连续内存数组中；

- [ ] 188、cvFindContours：从二值图像中寻找轮廓；

- [ ] 189、cvDrawContours：绘制轮廓；

- [ ] 190、cvApproxPoly：使用多边形逼近一个轮廓；

- [ ] 191、cvContourPerimeter：轮廓长度；

- [ ] 192、cvContoursMoments：计算轮廓矩；

- [ ] 193、cvMoments：计算Hu不变矩；

- [ ] 194、cvMatchShapes：使用矩进行匹配；

- [ ] 195、cvInitLineIterator：对任意直线上的像素进行采样；

- [ ] 196、cvSampleLine：对直线采样；

- [ ] 197、cvAbsDiff：帧差；

- [ ] 198、cvWatershed：分水岭算法；

- [ ] 199、cvInpaint：修补图像；

- [ ] 200、cvGoodFeaturesToTrack：寻找角点；

- [ ] ![](https://nimg.ws.126.net/?url=http%3A%2F%2Fdingyue.ws.126.net%2F2021%2F0508%2F684eae2fj00qsqvjb002fd200tz00fqg00id009m.jpg&thumbnail=660x2147483647&quality=80&type=jpg)  
### 200
- [ ] 201、cvFindCornerSubPix：用于发现亚像素精度的角点位置；

- [ ] 202、cvCalcOpticalFlowLK：实现非金字塔的Lucas-Kanade稠密光流算法；

- [ ] 203、cvMeanShift：mean-shift跟踪算法；

- [ ] 204、cvCamShift：camshift跟踪算法；

- [ ] 205、cvCreateKalman：创建Kalman滤波器；

- [ ] 206、cvCreateConDensation：创建condensation滤波器；

- [ ] 207、cvConvertPointsHomogenious：对齐次坐标进行转换；

- [ ] 208、cvFindChessboardCorners：定位棋盘角点；

- [ ] 209、cvFindHomography：计算单应性矩阵；

- [ ] 210、cvRodrigues2：罗德里格斯变换；

- [ ] 211、cvFitLine：直线拟合算法；

- [ ] 212、cvCalcCovarMatrix：计算协方差矩阵；

- [ ] 213、cvInvert：计算协方差矩阵的逆矩阵；

- [ ] 214、cvMahalanobis：计算Mahalanobis距离；

- [ ] 215、cvKMeans2：K均值；

- [x] 216、cvCloneMat：根据一个已有的矩阵创建一个新矩阵；

- [ ] 217、cvPreCornerDetect：计算用于角点检测的特征图；

- [ ] 218、cvGetImage：CvMat图像数据格式转换成IplImage图像数据格式；

- [ ] 219、cvMatMul：两矩阵相乘