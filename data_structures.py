# LİST
notes = [1, 2, 3, 4]
type(notes)

names = ["a", "b"]

# listeler kapsayıcıdır, içerisinde sayı, string veya farklı listeler olabilir.
not_nam = [1, 2, "a", True, [1, 2, 3]]
not_nam[0]
not_nam[5]

# Listeler değiştirilebilir.
notes[0] = 99

# liateyi 0dan 4e kadar yazdırır.
not_nam[0:4]

# LİSTE METHODLARI
dir(notes)
len(notes)
len(not_nam)

# listenin sonuna eleman ekler.
notes
notes.append(100)

# pop: indexe göre siler.
notes.pop(0)

# insert: index ekler. Aşağıdaki de 2.indexe 99 yazar
notes.insert(2, 99)

# SÖZLÜK
dictionary = {"REG": "Regression",
              "LOG": "Logistic Regression"}
# Aynı zamanda sözlükde key i yazdığında value sini çağırır.
dictionary = {"REG": ["rmse", 10],
              "LOG": ["MSE", 20]}
dictionary["LOG"][1]

# KEY SORGULAMA
"REG" in dictionary

# KEY E GÖRE VALUE ERİŞMEK
dictionary["REG"]
dictionary.get("REG")

# VALUE DEĞİŞTİRMEK
dictionary["REG"] = ["YSA", 10]

# Tüm KEY lere, valu lara ulaşmak
dictionary.keys()
dictionary.values()

# Tüm çiftleri TUPLE halinde listeye çevirme
dictionary.items()

# KEY-VALUE Değerlerini Güncellemek
dictionary.update({"REG": 11})

# Aynı zamanda update ile yeni bir key valur değeri ekleyebilirsin
# Eğer girdiğin key değeri, sözlükde yoksa ekler en sona . varsa değiştirir.
dictionary.update({"RF": 10})

# TUPLE
t = ("John", "mark", 1, 2)
type(t)
t[0]
t[0, 3]

# SET
# Liste üzerinden set oluşturma
set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

# set1'de olup set 2 de olmayanlar
set1.difference(set2)
set1 - set2

# symmetric_difference(): iki kümede birbirlerine göre olmayanlar
set1.symmetric_difference(set2)
set2.symmetric_difference(set1)

# intersection(): iki kümenin kesişimi
set1 = set([1, 3, 5])
set2 = set([1, 2, 3])
set1.intersection(set2)
set2.intersection(set1)

# 2 kümenin kesişimi için & işaretde kullanılabilir
set1 & set2

# union():İki kümenin birleşimi
set1.union(set2)

# isdisjoint(): İki kümenin kesişimi boş mu?
set1 = set([7, 8, 9])
set2 = set([5, 6, 7, 8, 9, 10])
set1.isdisjoint(set2)
set2.isdisjoint(set1)

# issubset(): Bir küme diğer kümenin alt kümesi mi?
set1.issubset(set2)

# issuperset():bir küme diğer kümeyi kapsıyor mu?
set2.issuperset(set1)



