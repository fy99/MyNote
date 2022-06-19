## DATA
SIZE OF
- int: 4
- float: 4
- double: 8
- char: 1
- short: 2
- long: 4
- long long: 8
- long double: 16

SCANF
scanf("%d%d", &a, &b);

### 指针

&v \*p
&v - &v = 元素的个数 

- int \*p = ( & int ) or ( int[] )
```
int a = 100;  
int arr[3] = {2, 3, 4};  
int *parr[5] = {&a, &arr[0], &arr[1], &arr[2], &a};  
  
int *p = arr;  
int (*pa)[3] = &arr;  
int **ppa = parr;  
int *(*pp_arr)[5] = &parr;
```


int arr[10] = {0};
- arr         首元素地址
- &arr[0]  首元素地址
- &arr       全部元素的地址

- 数组指针
	 int (\*p_arr) [10] = &arr  
	 int (\*\*p) [10] = &p_arr
- 指针数组
	 int \*p_arr [10] = {i1,i2}    
	 - 二级指针数组指针
		 int \*(\*p) [10] = &p_arr

```
int a = 100;  
int b = 200;  
int *p = &a;  
int **pp = &p;  
  
int *p_arr[2] = {&a, &b};  
int *(*pp_arr)[2] = &p_arr;  
  
char arr_c[5] = {0};  
char (*arr)[5] = &arr_c;  
char (**pa)[5] = &arr;  
char (***paa)[5] = &pa;
```

```
int arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};  
int *p_arr = arr;  
int (*p)[10] = &arr;  
int i = 0;  
for (i = 0; i < 10; i++) {  
    printf("(*p)[i]:%d\n", (*p)[i]);  
    printf("*(*p+i):%d\n", *(*p + i));  
    printf("*(p_arr+i):%d\n", *(p_arr + i));  
}
```
```
void print1(int arr[3][4], int x, int y) {  
  
  
    for (int i = 0; i < x; i++) {  
  
        for (int j = 0; j < y; ++j) {  
            printf("%d ", arr[i][j]);  
        }  
        printf("\n");  
    }  
}  
  
void print2(int (*arr)[4], int x, int y) {  
	 for (int i = 0; i < x; i++) {  
	  
	    for (int j = 0; j < y; ++j) {  
	        printf("%d ", arr[i][j]);  
	        printf("%d ", *(arr[i] + j));  
	        printf("%d ", *(*(arr + i) + j));  
	        printf("%d ", (*(arr + i))[j]);  
	    }  
	    printf("\n");  
	}
}  
  
int main() {  
  
    int arr[3][4] = {{1, 2, 3, 4},  
                     {5, 6, 7, 8},  
                     {9, 0, 1, 2,}};  
    print1(arr, 3, 4);  
    print2(arr, 3, 4);  
  
}


```

```
int arr[10]={0};
int *p=arr;
int i =0
arr[i]=*(arr+i)=*(p+i)=p[i]
```

![[parr.png]]


```
int arr[3][4] = {{1, 2, 3, 4},  
                 {5, 6, 7, 8},  
                 {9, 0, 1, 2,}};  
  
int (*p_0)[4] = &(arr[0]);  
int (*p[3])[4] = {&(arr[0]), &(arr[1]), &(arr[2])};
```

### 函数指针

```
int add(int x, int y) { return x + y; }  
  void add2(int x, int y) {}  
  
  
int main() {  
	
    int (*pf)(int, int) =add;  
    void (*pff)(int, int) =add2;  
	int a = (*pf)(3,5)
}
```