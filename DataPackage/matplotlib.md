
### show
1. plt.rcParams["font.sans-serif"] = ['simhei']  run config params
2. plt.rcParams['axes.unicode_minus'] = False  中文负号
3. plt.title('中文标题')
4. **plt.xlabel("x轴数据名",fontsize=20)  **
5. plt.ylabel("y轴数据名")
6. plt.plot()
	- linewidth
7. **plt.xticks(x[::10], labels=[将标签替换为])**
	plt.yticks() 轴数据
8. lengend() 图例
	plt.plot(x,y,label="标记说明")
	plt.lengend(loc="upper right") location
9. plt.text(x,y,"文字"),在数据坐标上写入文字信息
10. plt.grid(True) 显示网格
	-  linestyle
	- color 
	- linewidth
	- axis
11. plt.gca [get current axes] spine
	**plt.ylim(0,2500)** 限制轴的
	```python	  
	ax = plt.gca()
	plt.ylim(0,2500) 'y的范围'
	
	ax.spines["bottom"].set_color(None/"#ff7700")  
	ax.spines["left"].set_position(('data', 0.0))
	ax.spines["left"].set_position(('axes', 0.5))
	```
12. 大小
    plt.rcParams["figure.figsize"]=(8.0,4.0) # 英寸  
	plt.rcParams["figure.dpi"]=300
### plot
按次序 线状图
1. plt.plot(x,y)
- 'gH-.' green H -
- color "red"/"#ff7700"
- alpha 透明度
- linestyle '-'/ '--'/ ':' /'-.'
- linewidth
- marker  标记点样式 . o x D(钻石) H(六角) s(正方) +
- markeredgecolor="#ff7700"
- markersize
- markeredgewidth
## Figure
### Make
plt.figure(
num: 图像编号或名称
figsize:大小乘dpi
dpi
facecolor:背景色
edgecolor:边框色
frame_on:bool 是否显示边框
)

### subplot
1. add_axes([比例 => x:左, y下, w:宽, h:高])->ax 
2. subplot()
3. subplots()
#### ax
ax.set_xlim
ax.get_ylim