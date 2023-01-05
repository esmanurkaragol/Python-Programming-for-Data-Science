# NUPY = Numerical Python
# verimli veri saklama (sabit tipde veri tutar) ve vektörel hesaplamalar yapabilir.
# hızlıdır. Daha az çabayla çok iş yapar.
# numpy kütüphanesini çağırmak istediğinde np yazmam yeterli,kısa yol oluşturdum.

import numpy as np
import seaborn as sns

a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []
for i in range( 0, len( a ) ):
    ab.append( a[i] * b[i] )
print(ab)

# aynı işlemi numpy da yapalım.
import numpy as np
a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b

#Numpy Array'i OLuşturma yani NDARRAY
import numpy as np
np.array([1,2,3,4,5])
type(np.array([1,2,3,4,5]))

#zeros(), girilen sayı adedi kadar sıfır oluşturur.
# tipi int olan 0 lardan oluşan array oluşturur.
np.zeros(10, dtype=int)

#randint(alt limit, üst limit, size=kaç tane seçim yapmak istiyorsun) eğer alt limit girmezsen 0 kabul eder.
np.random.randint(0,10, size=10)


#normal(), eğer belirli bir istatiksel dağılıma göre sayın üretmek istersek kullanırız.
#normal(ortalama, argüman(standart sapması), boyut bilgisi)

print(np.random.normal(10, (3,4)), "Talha")

#NUMPY ARRAY ÖZELLİKLERİ
#ndim(), boyut sayısını verir
#shape(), boyut bilgisini verir.
#size(), toplam eleman sayısını verir.
#dtype(),array veri tipi

a= np.random.randint(10, size=5)
a.ndim
a.shape
a.size
a.dtype

#RESHAPİNG (YENİDEN ŞEKİLLENDİRME), boyut değiştirme işlemleri yapılır

import numpy as np
np.random.randint(1, 10, size=9)
np.random.randint(1,10, size=9).reshape(3,3)
#veya
ar = np.random.randint(1, 10, size=9)
ar.reshape(3,3)

#index seçimi, veri yapıları içindeki verilere ulaşmak için kullanırız.
import numpy as np
a= np.random.randint(10, size=10)
#belirli indexe erişim
a[0]
# belirli index aralığına erişim
a[0:5]
#ilgili indexdeki veriyi değiştirme
a[0] = 999

m = np.random.randint(10, size =(3,5))

# m çok boyutlu bir liste bunun indexlerine erişebilmek için m[satır, sütun] bilgisini verir.
#1.satır 1.sütundaki değere eriş
m[1,1]

m[2,3] = 999

#numpy tek veri tipinde tuttuğu için floatı int çevirerek tutar.
m[2, 3] = 99.9
#tüm satırları, 0.sütunu çeker
m[:, 0]
#1.saturu, tüm sutunları çeker
m[1, :]
# satırlardan da sütunlardan da belirli bir yere kadar veri çekebiliriz.
m[0:2, 0:3]

##FANCY INDEX, numpy arrayıne bir liste girildiğinde bize seçim işlemi sağlar.
#arrange(), belirli bir adım boyunca array oluşturur.
import numpy as np
v = np.arange (0,30,3)
v[1]

catch = [1, 2, 3]
v[catch]

#NUMPY DA KOŞULLU İŞLEMLER
import numpy as np
v = np.array([1, 2, 3, 4, 5])

#kalsik döngü ile array içeriisndeki elemanlardan 3 den küçük olanlara erişelim.
ab = []
for i in v:
    if i < 3 :
        ab.append(i)
#aynı işlemi numpy ile gerçekleştirme
v < 3

v[v < 3]
v[v != 3]

#MATEMATİKSEL İŞLEMLER
import numpy as np

v = np.array([1,2,3,4,5])
#tüm elemanları tek tek gezer ve ilgili işlemi her birine yapar
v / 5
v *5 /10

#aynı işlemleri methodlar aracılığıyla da yapabiliriz.
#çıkarma
np.subtract(v, 1)
#toplama
np.add(v, 1)
#ortalama
np.mean(v)
#toplam alma
np.sum(v)
np.min(v)
np.max(v)
#varyans
np.var(v)

#yapmış olduğun bu işlemlerin kalıcı olmasını istiyorsan tekrardan atama yapman lazım
v = np.subtract(v,1)

#NUMPY ile iki bilinmeyenli denklem çözümü
#1.değişken= xo; 2.değişken x1 kabul edelim.
#5*x0 + x1 = 12
#x0 + 3*x1 = 10

a= np.array(([5,1], [1,3]))
b= np.array([12,10])

#numpy da linalg modülünün içerisnde yer alan solve methodu çağrılarak bilinmeyen değerler hesaplanır.
np.linalg.solve(a,b)

#PANDAS: veri manipilasyonu veri analizi denildiğinde akla gelmesi gerekiyor.
#PANDAS DataFRAME:Çok boyutlu ve index bilgisi barındıran veri tipi.
#PANDAS SERİES: tek boyutlu ve index bilgisi barındıran veri tipi.
#pd serilerinde index bilgileri iç özellik olarak kalır. bu nedenle zamandan kazanç

import pandas as pd
#series() ile farklı tipte bir veri girdiğinde "pandas seri" sine çevirir
s= pd.Series([110,77, 12, 4, 5])
type(s)
s.index
s.dtype
s.size
#pandas serileri tek boyutludur.
s.ndim
s.values
type(s.values)
#head(x), ilk x veriyi getirir.
s.head(3)
#tail(x), sondan x veriyi getirir.
s.tail(3)

###########DIŞ KAYNAKLI VERİLERİ NASIL OKUYABİLİRİZ?
#VERİ OKUMA

import pandas as pd
df = pd.read_csv( "datasets/advertising.csv" )
df.head()

#VERİYE HIZLI BAKIŞ
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
print(df)
df.head()
#sondan değerlere göz atma
df.tail()
#boyut bilgisi
df.shape
#değişkenler ve değişkenlerin tipleri
df.info()
#değişkenlerin isimlerine erişmek için
df.columns
#index bilgisine erişmek istersek
df.index
#elimizdeki date frame in özet bir şekilde hızlıca istatistiklerine ulaşmak istiyorsak
df.describe()
#daha okunabilir olması için sonuna.T ekle
df.describe().T

#veri setinde eksiklik var mı? isnull() methodu kullan
df.isnull()

#verilerin herhangi birinde eksiklik var mı
df.isnull().values.any()

#verilerdeki değişiklik incelenmek istenirse. Hangi değişkende kaç tane eskiklik var.
#true=1, false=0 kabul eder
df.isnull().sum

#listelenen kategorik değişkenlerden birisinin kaç tane sınıf olduğu ve bu sınıfların kaçar tane olduğu bilgisine erişmek istiyorum.
#df["değişken ismi"]
df["sex"].head()
df["sex"].value_couunts()

#PANDAS' TA SEÇİM İŞLEMLERİ
import pandas as pd
import seaborn as sns
df= sns.load_dataset("titanic")
df.head()
#titanic verisinin indexlerine gitmek istiyorum
df.index
#slice işlemi yapmak için
df[0:13]

#indexlerde silme işlemi
#drop(silmek istediğin index, satırlardan mı sutunlardan mı sileceksin)
df.drop(0, axis = 0)
#yaptığın işlemi gözlemlemek adına head() at
df.drop(0, axis = 0).head()
#birden fazla indexi silmek istediğinde
delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis=0).head(10)
#silme işlemini kalıcı hale getirmek için bir değişkene tekraar ata
df = df.drop(delete_indexes, axis=0).head(10)
# inplace argümanını kullanarak değişikliği kalıcı hale getirebiliriz.
df.drop(delete_indexes, axis=0, inplace=True)

#Değişkeni İndexe Çevirme
df["age"].head()
df.age.head()

df.index = df["age"]

#silmek için:
df.drop("age", axis = 1).head()
#kalıcı olarak silmek için
df.drop("age", axis=1, inplace = True)

#df içine yazdığın veri eğer df in içinde yoksa içine eklenir.
df["age"]

df.resef_index().head()
df = df.reset_index()
df.head()

###DEğişkenler üzerinde işlemler

import pandas as pd
import seaborn as sns
#verilerin konsolda 3 noktali bir şekilde gözükmesini istemiyorsan bunu kullan
pd.set_option("display.max_colums", None)
ddf = sns.load_dataset("titanic")
df.head()
#bu veri seti içerisinde age var mı?
"age" in df

df["age"].head()
df.age.head()

#bir değişken seçtiğimizde veri tipi series olarak çıktı verebilir
df["age"].head()
type(df["age"].head())
#iki köşeli parantez içerisinde veriye baktığımızda veri bozulmaz df olarak kalır.
type(df[["age"]].head())

#bir df içerisinden birden fazla seçim yapmak istersek
df[["age", "alive"]]

#liste içeriisndeki değişkenleri çağırmak için
col_names = ["age", "adult_male", "alive"]

#df değişken eklemek istiyorsak,, köşeleri parantez içine olmayan bir veri yazarsan eklenir.
df["age2"] = df["age2"]**2
df["age3"] = df["age"] / df["age2"]

#silmek istediğimizde
df.drop("Age3", axis =1).head()
#loc = seçim yapmak için kullanılır
#(:, ...) bütün satırları seç.
#contains methodu, veilen stringi kendinden çnce var mı diye arar.
df.loc[:,df.columns.str.contains("age")].head()

#bunun dışındakilerini seçmek istiyorsan ~ işareti konulur
df.loc[:,~df.columns.str.contains("age")].head()

#iloc = integer based selection bilgisi vererek (index) seçim yapma işlemini sağlar.
#loc = label based selection.indexdeki labellara göre seçim yapar
impoer pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns. load_dataset("titanic")
df.head()
#o dan 3 e kadar seçim yapar(3dahil değildir burada 0. ,1. ve 2.indexleri alır)
df.iloc[0:3]
#df.ilpc[x.satıraki, y.sütündaki] elemanı çeker.
df.iloc[0,3]

#loc da 0 dan 3 e kadar seçim yaptığında 3 dahildir.
df.loc[0:3]

#verileirn adlarını kulllanark sütünları çekmek istiyorum
df.iloc[0:3, 0:3]
df.loc[0:3,, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]

#KOŞULLU SEÇİM
pd.set_option("display.max_columns", None)
df= sns.load_dataset("titanic")
df(head)

#ilgili veri setinde yaşı 50 den büyük olanlara ulaşalım
df[df["age"] > 50].head()
#yaşı 50den büyük toplam kaç kişi var,count.
df[df["age"] > 50]["age"].count()
#koşul + 2 tane sütun seçtik
df.loc[df["age"] > 50, "class"].head()
#birden fazla koşul gireceksen parantez içeriisne almalısın
#df.loc[(df["age"] > 50)
#       & (df["sex"] == "male")
#       &(df["embark_town] == "Cherbourgh") , ["age", "class", "embark_town]].head()

#TOPLULAŞTIRMA VE GRUPLAŞTIRMA FONKSİYONLARI
#toplulaştırma fonksiyonları

#count()
#first()
#last()
#mean()
#median()
#min()
#max()
#std()
#var()
#sum()
#pivot table

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

#yaşın ortalamasına erişmek istiyorum
df["age"].mean()
#cinsiyete göre yaşıın ortalamasına erişmek istediğimde, ilk olarak cinsiyete göre groupby işlemi yapılır.
df.groupby("sex")
#ardınan hesaplanmak istenilen verinin ortalamasını alacağımız fonksıyon yazılır.
df.groupby("sex")["age"].mean()
#cinsiyetin groupby aldıktan sonra agg ifadesi ise yaşların ortalamasını al
df.groupby("sex").agg({"age": "mean"})
#yukarıda yazmıs olduğumuz son 2 kod aynı şeyi ifade ediyor. bir üst satırda yazılan kodu tercih et.


df.groupby("sex").agg({"age": ["mean", "sum"],
                       "embark_town" : "count"})

#pıvot table oluşturma
import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()
df.pivot_table("survied","sex", "embarked")
df.pivot_table("survied", "sex", "embarked",aggfunc="std")  #aggfunc="std" standart sapmayı hesaplar

#sütünlar 2 boyutlu olsun istiyorsan 2 değişkeni liste içerisinde gir.
df.pivot_table("survied", "sex", ["embarked", "class"])

#sayısal değişkenleri kategorik değişkenlere çevirmek istiyorsam cut veya qcut fonksiyonları kullanılır.
#eğer veriyi tanıyorsan cut kullan, ama veriyi tanımıyorsan qcut kullan.
#qcut verileri otamatıkmen küçükten büyüğe sıralar. yüzdelik çeyrek durumlarına göre gruplara böler.
#cut(neyi böleceğini gir, neler böleceğini söyle)
df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])
df.pivot_table("survied", "sex", "new_age")

#Apply = Döngülerden daha pratikdir, satır ve sütünlarda otomatikmen çalışır
# LAMBDA = fonksiyon tanımlamadan, kullan at fonksiyonu yazar.
import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df= sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"]*2
df["age3"] = df["age"]*5

#değişkenleri 10 a bölmek istiyorsam
(df["age2"]/ 10).head()

#yukarıdaki kod pratik değil, döngü kullanalım
for col in df.columns:
    if "age" in col:
        print( (df[col] / 10).head() )
for col in df.columns:
    if "age" in col:
        df[col] = df[col]/ 10
df.head()

#Aynı işlemi lambda ve apply kullanarak yapalım
df[["age", "age2", "age3"]].apply(lambda x: x/10).head()
df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

#Birleştirme (JOİN) İşlemleri
import numpy as np
import pandas as pd
# 1ile 30 arasında ratgele 5-3 lük bir numpy arrayi oluşturduk
m = np.random.randint(1, 30, size=(5,3))
df1=pd.DataFrame(m, columns = ["var1", "var2", "var3"])
df2 = df1 +99

#2df alt alta birleştirmek istiyorum.
pd.concat([df1,df2])

#her iki df index bilgisini sıfırlamak için ignore_index = True yapmalısın.
pd.concat([df1,df2], ignore_index = True)

#MERGE ile birleştirme işlemleri
df1 = pd.DataFrame({"employees": ["john", "dennis", "mark", "maria"],
                    "group": ["accounting", "engineering", "engineering","hr"]})

df2 = pd.DataFrame({"employees": ["mark", "john", "dennis", "maria"],
                    "start_date": [2010, 2009, 2014, 2019]})
pd.merge(df1,df2)
#eğer birleştirmeyi özel bir veri ile yapmak istiyorsan on ile tanımla.
pd.merge(df1, df2, on="employees")

#Amaç her çalışanın müdürünün bilgisine erişmek
df3 = pd.merge(df1, df2)
df4 = pd.DataFrame({"group": ["accounting", "engineering", "hr"],
                   "manager": ["Caner", "Mustafa", "Berkcan"] })
pd.marge(df3, df4)
