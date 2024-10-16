def unique(list1):
    list2 = []
    for i in list1:
        if i not in list2:
            list2.append(i)
    return list2


list1 = [1, 2, 3, 4, 5, 4, 5, 3, 3, 3, 1]
list2 = unique(list1)
print(list2)