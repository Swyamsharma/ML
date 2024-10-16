arr1 = [1,2,3,4,5,6]
arr2 = [6,7,8,9,10]
concatenated_arr = arr1 + arr2
sorted_arr1 = sorted(arr1)
sorted_arr2 = sorted(arr2)
added_arr = [x + y for x, y in zip(arr1, arr2)]
subtracted_arr = [x - y for x, y in zip(arr1, arr2)]
multiplied_arr = [x * y for x, y in zip(arr1, arr2)]
divided_arr = [x / y for x, y in zip(arr1, arr2)]
print("Concatenated array:", concatenated_arr)
print("Added array:", added_arr)
print("Subtracted array:", subtracted_arr)
print("Multiplied array:", multiplied_arr)
print("Divided array:", divided_arr)