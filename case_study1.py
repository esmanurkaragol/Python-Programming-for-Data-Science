#GÖREV1
x= 8
type(x)

y= 3.2
type(y)

z=8j+18
type(z)

a= "hello world"
type(a)

b = True
type(b)

c= 23<22
type(c)

l = [1,2,3,4]
type(l)

d= {"Name" : "Jake",
    "Age": 27,
    "Adress": "Downtown"}
type(d)

t=("Machine Learning", "Data Science")
type(t)

s= {"python", "Machine Learning "}
type(s)

#GÖREV2
#upper();karakterleri büyük harfe çevirir.
#replace(); karakterleri değişitrmek için
#split(): bölmek için kullanılır

text = "The goal is to turn data into information, and information into insight"
text = text.upper()
new_text = text.replace("," , " ").replace("." , " ").split()
print(new_text)

#GÖREV3
list = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

#adım1
len(list)

#adım2
list[0]
list[10]

#adım3
new_list = list[0:4]
print(new_list)

#adım4
#pop(): belirtilen bir indeksdeki elemanı siler.
list = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]
list.pop(8)
print(list)

#adım5
list.append("case")
print(list)

#adım6
#ınsert(): belirtilen indexe eleman ekler.
list.insert(8, "N")
print(list)

#GÖREV4
dict ={"Christian": ["America", 18],
       "Daisy": ["England", 12],
       "Antonio": ["Italy", 25]}
#adım1
#.keys(); tüm key değerlerini yazar.
print(dict.keys())

#adım2
#.values(); tüm value yazar.
print(dict.values())

#adım3
dict["Daisy"] = ["England", 13]
print(dict)

#adım4
dict["Ahmet"] = ["Turkey", 24]
print(dict)

#adım5
#pop(): ilgili veriyi siler.
dict.pop("Antonio")
print(dict)

#GÖREV5
l = [2, 13, 18, 93, 22]
def func(list):
    even_list = []
    odd_list = []

    for i in list:
        if i %2 == 0:
            even_list.append(i)
        else:
            odd_list.append(i)
    return even_list, odd_list

even_list, odd_list = func(l)

#GÖREV6(index numarasının 1 den başlaması gerekiyordu)
muhendislik = ["Ali", "Veli", "Ayşe"]

for i,x in enumerate(muhendislik):
    print("Mühendislik Fakültesi", i, ". öğrenci",x  )

tıp = ["Talat","Zeynep", "Ece"]
for i,x in enumerate(tıp):
    print("Tıp Fakültesi", i, ". öğrenci",x)

#GÖREV7
ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3,4,2,4]
kontenjan = [30,75,150,25]
for ders_kodu, kredi, kontenjan in zip(ders_kodu,kredi, kontenjan):
    print(f"Kredisi {kredi} olam {ders_kodu} kodlu dersin kontenjanı {kontenjan} kişidir.")



#GÖREV8
kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])
if kume1.issuperset(kume2):
    print(kume1.intersection(kume2))
else:
    print(kume2.difference(kume1))



