# 函数指针

```
int add(int x, int y) { return x + y; }  
  void add2(int x, int y) {}  
  
  
int main() {  
	
    int (*pf)(int, int) =add;  
    void (*pff)(int, int) =add2;  
	int a = (*pf)(3,5)
}
```