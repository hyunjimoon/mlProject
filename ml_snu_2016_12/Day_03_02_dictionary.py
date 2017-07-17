# Day_03_02_dictionary.py

# key, value
# a = {'color': 'red', 'price': 100}
a = dict(color='red', price=100)
print(a)
print(a['color'], a['price'])

a['title'] = 'pen'      # insert
print(a)

a['title'] = 'potato'   # update
print(a)

for key in a:
    print(key, a[key])

print(a.items())
b = list(a.items())
print(b)

for i in b:
    print(i, i[0], i[1])

# a1, a2 = (1, 2)
# print(a1, a2)
for k, v in b:
    print(k, v)

for i in a.items():
    print(i, i[0], i[1])

for k, v in a.items():
    print(k, v)


