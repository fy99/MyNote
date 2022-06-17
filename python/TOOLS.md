## Hotkey
```PYTHON
import os  
import time  
  
import keyboard  
  
keys = [("ctrl", "alt", "a"), "f10"]  
  
keyboard.add_hotkey(keys[0], lambda: os.system(r"C:\Users\fy\Desktop\python.py\ScreenShot\QQSnapShot.exe"))  
keyboard.add_hotkey(keys[1], lambda: os.system(r"C:\Users\fy\Desktop\python.py\ScreenShot\QQSnapShot.exe"))  
# keyboard.remove_hotkey("f10")  
  
try:  
    input("F10 Ctrl+Alt+A")  
finally:  
    keyboard.remove_hotkey(keys[1])  
    keyboard.remove_hotkey(keys[0])  
    print("removed hotkey")
```

## ThreadPool
```python
from multiprocessing.pool import ThreadPool  
import multiprocessing  
import random  
import time  
  
  
def d(a, b=None):  
    if b:  
        print(f"{b=}")  
    d = random.random()  
    d *= 3  
    print(f"start:{a}:{d=}")  
      
    time.sleep(d)  
    print(f"end:{a}:{d=}")  
    return d  
  
  
if __name__ == '__main__':  
    # from multiprocessing.pool import ThreadPool  
    m = ThreadPool(10)  
    # m = multiprocessing.Pool()  
    """  
    apply(function=fun,args=(a,),**kwargs={"a":1})    
    map; 只解包一次，里面无论数组列表均当做第一个参数传入  
    async has callback    apply_async():callback 函数完成后直接运行callback函数  
    map_async:等所有函数完成后，把所有结果组织为一个列表，然后把列表当成参数传入callback函数  
    starmap: *map 可以解包多个参数，然后分别传入  
	chunksize 把任务按一定数分组，一个线程/进程一次放入多个任务  
    imap    等待任务完成后在创建新任务，开始时任务参数只保存在列表中，
	    占用更少的内存，但更慢，像没有展开的 range();没有callback  
        应该把chunksize设置大点 任务列表用yield生成任务  
    imap_unordered; unordered    
    """    
    # m.map_async(d,range(100),callback=lambda r:print(f"back:{r=}"),chunksize=5)  
    # m.map(d,range(100),chunksize=10)        # m.map(d,[(i,i+1) for i in range(100)],chunksize=10)  
        # m.imap(d,range(100),chunksize=5)  
    # m.imap_unordered(d,list(range(100)),chunksize=1)        # for i in range(100):  
    #   m.apply(d,(i,))    #   m.apply_async(d,(i,),callback=lambda p:print(f"call_back:{p}"))        # m.starmap(d,[(i,i+1) for i in range(100)],chunksize=10)  
    # m.starmap_async(d,[(i,i+1) for i in range(100)],chunksize=10,callback=lambda p:print(f"call_back:{p}"))          
	m.close()  
    m.join()
```

## 单一按键
```python
import msvcrt
i = msvcrt.getch()        .decode().lower()
```