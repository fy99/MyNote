## Infinite iterators
1. count(start,[step]) 
	- count(100,5) => 100 105 110 ...
2. cycle()
	cycle("abcd") => a b c d a b c d a b c d ...
3. repeat() 带次数可以终止
	repeat("abcd") => abcd abcd abcd ...
	repeat("abcd", 2) => abcd abcd


## ==Combinatoric iterators==

| | 元素可重复 | 元素不可重复 |
|:-:|:-:|:-:|
| 有序(更多)-排列 | product (最多) | permutations |
| 无序(去重)-组合| combinations_with_replacement | combinations (最少) |

1. ==product== 元素可重复 有序
	- product('ABCD', repeat=2) 
		AA AB AC AD BA BB BC BD CA CB CC CD DA DB DC DD
		
2. ==permutations== (排列) 元素不可重复 有序
	- permutations('ABCD', 2)
		AB AC AD BA BC BD CA CB CD DA DB DC
		
3. ==combinations== 元素不可重复 无序
	 combinations('ABCD', 2)
		AB AC AD BC BD CD
		
4. ==combinations_with_replacement== 元素可重复 无序
	combinations_with_replacement('ABCD', 2)
		AA AB AC AD BB BC BD CC CD DD


## Iterator
1. accumulate([1,2,3,4,5]) --> 1 3 6 10 15  累积
2. chain('ABC', 'DEF') --> A B C D E F
3. chain.from_iterable(['ABC', 'DEF']) --> A B C D E F
4. compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
5. dropwhile(lambda x: x<5, [1,4,6,4,1]) --> 6 4 1
6. filterfalse(lambda x: x%2, range(10)) --> 0 2 4 6 8
7. groupby()
8. islice('ABCDEFG', 2, None) --> C D E F G
9. pairwise('ABCDEFG') --> AB BC CD DE EF FG
10. starmap(pow, [(2,5), (3,2), (10,3)]) --> 32 9 1000
11. takewhile(lambda x: x<5, [1,4,6,4,1]) --> 1 4
12. tee()
13. zip_longest('ABCD', 'xy', fillvalue='-') --> Ax By C- D-