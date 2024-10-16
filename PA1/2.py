a = 5
b = 6
a, b = b, a
print(a, b)
temp = a
a = b
b = temp
print(a, b)