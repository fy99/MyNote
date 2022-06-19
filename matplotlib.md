
### show
1. plt.rcParams["font.sans-serif"] = ['simhei']  run config params
2. plt.rcParams['axes.unicode_minus'] = False  中文负号
3. plt.title('中文标题')
4. plt.xlabel("x轴数据名",fontsize=20)  
5. plt.ylabel("y轴数据名")
6. plt.plot()
	- linewidth
7. plt.xticks(x[::10], labels=[将标签替换为])  
	plt.yticks() 轴数据
8. lengend() 图例
	plt.plot(x,y,label="标记说明")
	plt.lengend(loc="upper right") location
9. plt.text(x,y,"文字"),在数据坐标上写入文字信息


### plot
按次序 线状图
1. plt.plot(x,y)
