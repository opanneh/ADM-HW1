# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 11:44:48 2020

@author: panne
"""



"""py_introduction"""

#print Hello World
my_string = "Hello, World!"
print(my_string)



#If_else
n = int(input())

if n%2==1:
    print("Weird")

else:
    if n%2==0 and (n in range(2,6) or n>20):
        print("Not Weird")

    else:
        if n%2==0 and (n in range(6,21)):
            print("Weird")


#arithmetic operators
a = int(input())
b = int(input())

print(a+b)
print(a-b)
print(a*b)


#Division
a = int(input())
b = int(input())

print(a//b)
print(a/b)


#loops
n = int(input())

for n in range(0,n):
    print(n**2)

#write a function
def is_leap(year):
    leap = False
    
    # Write your logic here
    if year%4==0:
        if (year%100==0 and not year%400==0):
            leap = False
        else:
            leap =True
    
    return leap

year = int(input())
print(is_leap(year))


#print a function
n = int(input())

for i in range(1,n+1):
    print (i, end='')
    


"""Basic Data Types"""

#List Comprehensions
x = int(input())
y = int(input())
z = int(input())
n = int(input())

print([[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if sum([i,j,k]) != n])


#runners up
n = int(input())

arr = map(int, input().split())
print (sorted(set(arr))[-2])


#nested lists
n=int(input())

stu_name=[]
stu_score=[]

for i in range(n):
    stu_name.append(input())
    stu_score.append(float(input()))

x=min(stu_score)
score_list=[i for i in(stu_score) if i!=x]

x=min(score_list)
name_list=[stu_name[j] for j in range(n) if x==stu_score[j]]

n=len(name_list)
name_list.sort()

for i in range(n):
    print(name_list[i])



"""Strings""" 

#Swap Case
def swap_case(s):   
    return s.swapcase()


if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)
    

#What's your name
def print_full_name(a, b):
    print("Hello {} {}! You just delved into python.".format(first_name, last_name))


if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)
    
#mutations
def mutate_string(string, position, character):
    return string[:position] + character + string[position + 1:]

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)
    

#find a string
def count_substring(string, sub_string):
    count = 0
    for i in range(len(string)):
        for j in range(i, len(string)+1):
            if(string[i:j]==sub_string):
                count += 1
    return count

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)
    

#string Valllidators
string_s = input()

my_list=list(string_s)

alp_n,alp_c,alp_digits,any_upper,any_lower=False,False,False,False,False

for i in my_list:
    if i.isalnum():
        alp_n=True
    if i.isalpha():
        alp_c=True
    if i.isdigit():
        alp_digits=True
    if i.islower():
        any_lower=True
    if i.isupper():
        any_upper=True
print (alp_n)
print (alp_c)
print (alp_digits)
print (any_lower)
print (any_upper)


#Text alignment
#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))
    

#Text Wrap
import textwrap

def wrap(string, max_width):
    return textwrap.fill(string, max_width)

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

#Door Mat
if __name__ == '__main__':
        N, M = map(int, input().split(" "))

        for i in range(N):
                pattern = ".|."
                if i < (N-1)/2:
                        print((pattern * (2*i+1)).center(M, "-"))
                elif i == (N-1)/2:
                        print("WELCOME".center(M, "-"))
                else:
                        print((pattern * (2*(N-1-i)+1)).center(M, "-"))


#String formatting
def print_formatted(number):
    # your code goes here
    width = len(str(bin(n)))-2
    for num in range(1, n+1):
        decimal = int(num)
        octal = oct(num)
        hexadecimal = hex(num)
        binary = bin(num)

        print(str(decimal).rjust(width), octal[2:].rjust(width), hexadecimal[2:].title().rjust(width), binary[2:].rjust(width))

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)
    


"""Sets"""

#Intro to sets
def average(array):
    # your code goes here
    return sum(set(array)) / len(set(array))

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)


#No Idea
params = input()
arr = input().split()
A = set(input().split())
B = set(input().split())

outcome=0

for i in arr:
    if (i in A):
        outcome+=1
    if (i in B):
        outcome-=1
        
print (outcome)


#Symetric Difference
input()
s1=set(map(int, ((input().split()))))
input()
s2=set(map(int, ((input().split()))))
sym = sorted(s1 ^ s2)
for i in sym:
    print(i)


#set add
n = int(input())
s = set()
for i in range(n):
    s.add(input())
print (len(s))

#set discard
n = int(input())
s = set(map(int, input().split()))
for i in range(int(input())):
    command =  input().split()
    if command[0] == 'pop':
        s.pop()
    elif command[0] == 'remove':
        s.remove(int(command[1]))
    elif command[0] == 'discard':
        s.discard(int(command[1]))
print(sum(s))

#set union
n = input()
set_n = set(map(int, input().split()))
b = input()
set_b = set(map(int, input().split()))
print(len(set_n.union(set_b)))

#set intersection
set1 = int(input())
setline1 = set(map(int,input().split()))
set2 = int(input())
setline2 = set(map(int,input().split()))
print (len(setline1.intersection(setline2)))


#set difference
n1 = int(input())
set_1 = set(map(int,input().split()))
n2 = int(input())
set_2 = set(map(int,input().split()))
print(len(set_1-set_2))


#Set Symetric Difference
n1 = int(input())
set1 = set(input().split())

n2 = int(input())
set2 = set(input().split())

result = set1.union(set2) - set1.intersection(set2)
print(len(result))


#Sets Mutations
m=int(input())
set1=set(map(int,input().split()))
n=int(input())
for i in range(n):
    cmd=input().split()
    set2=set(map(int,input().split()))
    eval('set1.{0}({1})'.format(cmd[0],set2))
print(sum(set1))


#sets captains room
d=input() 
a=input().split()  
set1=set() 
set2=set()  
for i in a:
    if  i in set1:
        set2.add(i)
    else:
        set1.add(i)
set3=set1.difference(set2)
print (list(set3)[0])


#Set subset
n = int(input())

for i in range(n):
    a = int(input())
    set_a = set(input().split())
    b = int(input())
    set_b = set(input().split())
    out_set = set_a.difference(set_b)
    if len(out_set) == 0:
        print(True)
    else:
        print(False)
        

#Set Strict Super Sets
a = set(input().split())
counter , n = 0, int(input())
for i in range (n):
        b = set(input().split())
        if a.issuperset(b) :
                counter += 1
print(counter == n)


"""Math"""

#Polar coordinates
import cmath

r = complex(input().strip())

print(cmath.polar(r)[0])
print(cmath.polar(r)[1])


#Angle MBC
import math

ab = int(input())
bc = int(input())

hypc = math.sqrt((ab**2)+(bc**2))
acos = ((bc*bc) + (hypc*hypc) - (ab*ab)) / (2*(bc*hypc))
ans = math.degrees(math.acos(acos))

print (str(int((round(ans)))) + '°')


#Tringle Quest 2
for x in range(1,int(input())+1):
    print(((10**x - 1)//9)**2)
    

#did mod
a, b = int(input()), int(input())

print(a//b)
print(a%b)
print(divmod(a,b))


#power mod power
a, b, m = int(input()),int(input()), int(input())

print(pow(a,b))
print(pow(a,b,m))


#Integers comes in all sizes
a,b,c,d = int(input()), int(input()), int(input()), int(input())

print((a ** b) + (c ** d))


#Triangle Quest 1
for i in range(1,int(input())): #More than 2 lines will result in 0 score. Do not leave a blank line also
    print(i * (10**i - 1)//9)
    

"""Itertools"""

#product
from itertools import product

a = map(int, input().split())
b = map(int, input().split())

print(*product(a, b))


#permutations
from itertools import permutations as pm

word, num = input().split(" ")
pm = list(pm(word, int(num)))
pm.sort()

for i in pm:
    print("".join(i))
    

#combinations
from itertools import combinations as cm

s , n  = input().split()

for i in range(1, int(n)+1):
    for j in cm(sorted(s), i):
        print (''.join(j))
        

#combinations with replacement
from itertools import combinations_with_replacement as cwr

s, k = input().split()

for c in cwr(sorted(s), int(k)):
    print("".join(c))

#compress string
from itertools import groupby
for k, g in groupby(input()):
    print("({}, {})".format(len(list(g)), k), end=" ")
    

#Iterable Iterators
from itertools import combinations as cm

N = int(input())
a = input().split()
K = int(input())
c = list(cm(a,K))

result = [1 for i in range(len(c)) if 'a' in c[i]]
print(sum(result)/len(c))


#Maximize it
K, M = [int(x) for x in input().split()]
arrayN = []
for _i_ in range(K):
    arrayN.append([int(x) for x in input().split()][1:])
    
from itertools import product
p_combinations = list(product(*arrayN))

def func(nums):
    return sum(x*x for x in nums) % M

print(max(list(map(func, p_combinations))))




"""Collections"""

#Collection counter
import collections as cln

numShoes = int(input())
shoes = cln.Counter(map(int, input().split()))
numCust = int(input())

income = 0

for i in range(numCust):
    size, price = map(int, input().split())
    if shoes[size]: 
        income += price
        shoes[size] -= 1

print (income)


#Default Dict
from collections import defaultdict as dd
d = dd(list)

n, m = map(int, input().split())

for i in range(1,n+1):
    d[input()].append(str(i))
    
for i in range(m):
    print (' '.join(d[input()]) or -1)


#NamedTuple
from collections import namedtuple as nt

length = int(input())
Student = nt('Student', ' '.join(input().split()))
marksSum = 0

for i in range(length):
    inputs = input().split()
    a = Student._make(inputs)
    marksSum += int(a.MARKS)

print(round(marksSum/length, 2))


#OrderedDict
from collections import OrderedDict as OD

number_ = int(input())
odict = OD()

for i in range(number_):
    litem = input().split(' ')
    price = int(litem[-1])
    item_name = " ".join(litem[:-1])
    
    if odict.get(item_name):
        odict[item_name] += price
    else:
        odict[item_name] = price

for i,v in odict.items():
    print(i,v)


#Word order
from collections import OrderedDict
words = OrderedDict()

for _ in range(int(input())):
    word = input()
    words.setdefault(word, 0)
    words[word] += 1
   
print(len(words))
print(*words.values())


#Deque
from collections import deque

if __name__ == '__main__':
        n = int(input())
        que = deque()
        for i in range(n):
                stmts = input().split()
                if len(stmts) == 2:
                        getattr(que, stmts[0])(int(stmts[1]))
                else:
                        getattr(que, stmts[0])()
        print (' '.join(map(str, que)))
        



"""Date and Time"""

#Calender Module
import calendar
m, d, y = map(int, input().split())
print(calendar.day_name[calendar.weekday(y, m, d)].upper())



"""Errors and Exceptions"""

#Exceptions
for i in range(int(input())):
    try:
        a,b = map(int,input().split(' '))
        print (a//b)
    except (ZeroDivisionError,ValueError) as e:
        print ("Error Code:",e)
        

#Incorrect Regex
import re
n = int(input())
for i in range(n):
    string = input()
    try:
        regex = re.compile(string)
        print(True)
    except re.error:
        print(False)
        


"""Built-Ins"""

#Zipped
n, x = map(int, input().split())

sheet = []
for _ in range(x):
    sheet.append(map(float, input().split()))

for i in zip(*sheet):
    print(sum(i) / len(i))


#Input()
x, k = map(int, input().split())
print(eval(input()) == k)



#Evaluation
eval(input())

#Athlete Sort
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())


    arr.sort(key=lambda arr:arr[k])
    for i in arr:
        s=' '.join(map(str,i))
        print(s) 


#Any and All
N,n = int(input()),input().split()
print (all([int(i)>0 for i in n]) and any([j == j[::-1] for j in n]))



"""Functionals"""

#Map $ Lambda
cube = lambda x: pow(x, 3) # complete the lambda function

fib = []

def fibonacci(n):
    # return a list of fibonacci numbers
    for i in range(n):
            if i == 0:
                fib.append(0)
            if i == 1:
                fib.append(1)
            if i > 1:
                fib.append(fib[i-1] + fib[i-2])
    return fib


#Vallidating Emails
import re

def fun(s):
    # return True if s is a valid email, else return False
    
    pattern = '^[a-zA-Z0-9_-]+@[a-zA-Z0-9]+[.]\w{1,3}$'
    return re.search(pattern, s)



#reduce function
from fractions import Fraction
from functools import reduce

def product(fracs):
    t = Fraction(reduce(lambda x,y:x*y,fracs)) 
    
    return t.numerator, t.denominator

if __name__ == '__main__':
    fracs = []
    for _ in range(int(input())):
        fracs.append(Fraction(*map(int, input().split())))
    result = product(fracs)
    print(*result)


"""Regex and Parsing"""

#Detect floats
import re
n = int(input())
for i in range(n):
    s = input()
    print(bool(re.search(r"^[+-]?[0-9]*\.[0-9]+$",s)))


#Re.split()
regex_pattern = r"[\.,A-Za-z]"	# Do not delete 'r'.

import re
print("\n".join(re.split(regex_pattern, input())))


#Groups
import re
m = re.search(r"([a-z0-9])\1+", input())
print(m.group(1) if m else -1)


#find all
import re
m = re.findall('(?=[^AEIOUaeiou]?)([AEIOUaeiou]{2,})(?=[^AEIOUaeiou])+',input())
print('\n'.join([item for item in m]) if m else -1)



#Re.Start
S = input()
k = input()
import re
pattern = re.compile(k)
r = pattern.search(S)
if not r: print ("(-1, -1)")
while r:
    print ("({0}, {1})".format(r.start(), r.end() - 1))
    r = pattern.search(S,r.start() + 1)
    

#Vallidate roman numerals
egex_pattern = r"^(?:([MCXI]){1,3}(?!\1)|([VLD])(?!\2))+$"	# Do not delete 'r'.

import re
print(str(bool(re.match(regex_pattern, input()))))


#Validate phone number
import re
N=int(input())

for i in range(N):

    if re.match(r'[789]\d{9}$',input()):   
        print ('YES')  
    else:  
        print ('NO') 
        
        
#Vallidate emails
import re
n = int(input())
for _ in range(n):
    x, y = input().split(' ')
    m = re.match(r'<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>', y)
    if m:
        print(x,y)
        
        
#Hex color
import re
fmt=r'(?<=.)#[0-9A-Fa-f]{3}\b|#[0-9A-Fa-f]{6}\b'
l=int(input())
for i in range(l):
    x=input()
    k=re.findall(fmt,x);
    for j in k:
        print(j)



"""XML"""

#find the score
import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    # your code goes here
    return len(node.attrib) + sum(get_attr_number(child) for child in node)

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))
    

#find max depth
import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    # your code goes here
    level += 1
    if level >= maxdepth:
        maxdepth = level
    for child in elem:
        depth(child, level)
        
if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)
    

"""Decorators"""

#std mobile numbers
 def wrapper(f):
    def fun(l):
        # complete the function
        f(["+91 "+c[-10:-5]+" "+c[-5:] for c in l])
    return fun



@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 
    


#Name Directory
import operator

def person_lister(f):
    def inner(people):
        # complete the function
        return [f(i) for i in sorted(people,key=lambda x:int(x[2]))]
    return inner


@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')
    


"""Numpy"""

#arrays

import numpy
def arrays(arr):
    # complete this function
    # use numpy.array

    return(numpy.array(arr[::-1], float))

arr = input().strip().split(' ')
result = arrays(arr)
print(result)


#shape and reshape
import numpy as np

arr=input().strip().split(' ')
arr=np.array(arr,int)
print(np.reshape(arr,(3,3)))


#Transpose
import numpy as np

r, c = map(int,input().split())
i = np.array([input().split() for _ in range(r)], int)
print (np.transpose(i))
print (i.flatten())


#concatenate
import numpy as np

n,m,p = input().split(" ")
k = []

for i in range((int(n)+int(m))):
    k.append([int(x) for x in input().split(" ")])
print(np.array(k))


#Zeros and ones
import numpy as np

N = tuple(map(int,input().strip().split()))
z = np.zeros(N, dtype=np.int)
o = np.ones(N, dtype=np.int)

print (z)
print (o)


#Eye and Identify
import numpy as np

np.set_printoptions(sign=' ')
print(np.eye(*map(int, input().split())))


#Array Math
import numpy as np

n, m = map(int, input().split())

a, b = (np.array([input().split() for _ in range(n)], dtype=int) for _ in range(2))

print(a+b)
print(a-b)
print(a*b)
print(a//b)
print(a%b)
print(a**b)


#Ceil_floor
import numpy as np

np.set_printoptions(legacy='1.13') 
A = np.array((input().split()), float)

print(np.floor(A))
print(np.ceil(A))
print(np.rint(A))


#sum and prod
import numpy as np

N, M = map(int, input().split())
A = np.array([input().split() for _ in range(N)],int)

print(np.prod(np.sum(A, axis=0), axis=0))


#min ans max
import numpy as np

N, M = map(int, input().split())
A = np.array([input().split() for _ in range(N)],int)

print(np.max(np.min(A, axis=1), axis=0))


#mean var std
import numpy as np 

n,m = map(int, input().split())
b = []
for i in range(n):
    a = list(map(int, input().split()))
    b.append(a)

b = np.array(b)

np.set_printoptions(legacy='1.13')
print(np.mean(b, axis = 1))
print(np.var(b, axis = 0))
print(np.std(b))


#Dot and Cross
import numpy as np

M =int(input())
array1 =np.array([input().split() for _ in range(M)],int)
array2 =np.array([input().split() for _ in range(M)],int)
np.set_printoptions(sign=" ")

print(np.dot(array1, array2 ))


#Inner Outer
import numpy as np

A=np.array(input().split(),int)
B=np.array(input().split(),int)

print(np.inner(A,B))
print(np.outer(A,B))



#Polynomials
import numpy as np

n = list(map(float,input().split()))
m = input()

print(np.polyval(n,int(m)))


#Linear Algebra
import numpy as np

n=int(input())
a=np.array([input().split() for _ in range(n)],float)
np.set_printoptions(legacy='1.13') 

print(np.linalg.det(a))





"""Birthday Cake Candles"""

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    # Write your code here
    candles.sort()

    result = candles.count(candles[len(candles)-1])
    return result

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()
    
    
    

"""Kangaroo/Number Line Jumps"""

import math
import os
import random
import re
import sys

# Complete the kangaroo function below.
def kangaroo(x1, v1, x2, v2):

    if x2 > x1 and v2 > v1:
            return "NO"
    else:
            if v2-v1 == 0:
                return 'NO'
            else:
                result = (x1-x2) % (v2-v1)
                if result == 0:
                    return 'YES'
                else:
                    return 'NO'

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()




"""Viral Adverts"""

import math
import os
import random
import re
import sys

# Complete the viralAdvertising function below.
def viralAdvertising(n):
        shared =5
        cumulative=0
        for i in range(1,n+1):
            liked = shared//2
            cumulative+=liked
            shared = liked*3
        return cumulative


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()


"""Recursive Digit Sum"""

import math
import os
import random
import re
import sys

# Complete the superDigit function below.
def superDigit(n, k):

    digits = map(int, list(n))
    return get_super_digit(str(sum(digits) * k))

def get_super_digit(p):
    if len(p) == 1:
        return int(p)
    else:
        digits = map(int, list(p))
        return get_super_digit(str(sum(digits)))


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()


"""INsertion Sort 1 """"

import math
import os
import random
import re
import sys

# Complete the insertionSort1 function below.
def insertionSort1(n, arr):

    target = arr[-1]
    idx = n-2
    
    while (target < arr[idx]) and (idx >= 0):
        arr[idx+1] = arr[idx]
        print(' '.join(map(str, arr)))
        idx -= 1
        
    arr[idx+1] = target
    print(' '.join(map(str, arr)))
    

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


"""Insertion 2 """"

import math
import os
import random
import re
import sys

# Complete the insertionSort2 function below.
def insertionSort2(n, arr):
    for i in range(n):
        if(i == 0):
            continue
        for j in range(0, i):
            if(arr[j] > arr[i]):
                temp = arr[i]
                arr[i] = arr[j]
                arr[j] = temp
            else:
                continue
        print(*arr)

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)




