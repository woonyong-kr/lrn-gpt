import json

ids = [260, 4, 10]
# ids - list[int]
# return => str형
ids_list = []

for i in range(len(ids)):
    # value = ids[i]
    # ids_list.append(bytes([value]))
    print("hello")

value = ids[0]

ids_list.append(bytes([value]))
ids_list.append(bytes([2]))
print(ids_list)