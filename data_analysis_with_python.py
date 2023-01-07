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


#VERİ GÖRSELLEŞTİRME: MATPLOTLİB SEABORN
#Matplot Library: düşük seviyeli veri görselleştirme aracıdır. veri görselleştirme yapmak için çok çaba gerektirir.
#powerBI gibi iş zekası araçları veri görselleştimeye daha uygundur.
#SEABORN:

######SÜTÜN GRAFİĞİ:
#Elinde kategorik değişken varsa kullan.
#Bunuyapmak için matplotlib de; countplot ile gerçekleştirilir.
#bunu yapmak için seaborn da; bar ile gerçekleştirilir.

######HİSTOGRAM (HİS) ve BOXPLOT(Kutu Grafik)
#sayısal değişkenler olduğunda bu iki grafik kullanılır.
#böylece veri dağılımını görmüş oluruz.


#Kategorik Değişken Görselleştirme
#value_counts(): ilgili kategorik değişkeni betimler.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
#plot dediğimde bunun çizdir demiş oluyorum aslında. ne tipde çizeceğinide kind= diyip tür bilgisini ver.
df["sex"].value_counts().plot(kind="bar")
plt.show() #grafiği ekrana yazdırmak istediğimde

#SAYISAL DEĞİŞKEN GÖRSELLEŞTİRME
#1.HİSTOGRAM GRAFİK KULLANARAK YAPALIM,sayısal değişkenlerinin dağılımını verir.
#plt.hist(görselleştirmek istediğin değişkeni yaz)
plt.hist(df["age"])
plt.show()

#2.BOXPLOT
plt.boxplot(df["fare"])
plt.show()

#MATPLOTLİB ÖZELLİKLERİ
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
pd.set_option("display.max_columb",None)
pd.set_option("display.width", 500)

#plot özelliği: veriyi görselleştirmek için kullandığımız fonksiyonlardan birisidir.
x=np.array([1,8])
y = np.array([8,150])
plt.plot(x,y)
plt.show()
#değerlerin olduğu yere nokta koymak için "o" değerini gir.
plt.plot(x, y, "o")
plt.plot(x,y)
plt.show()

#marker özelliği: işaretleyici özelliğidir.
y = np.array([13,28, 11, 100])
#diyelim ki y noktalarına daire koymak istiyorum yani marker ile işaretlemek istiyorum.
plt.plot(y, marker="o")
plt.show()

plt.plot(y, marker="*")
plt.show()

#markers tipleri;
#o, *, ., , , x, X, +, P, s, D, d, p,H,h

#line-çizgi özelliği
y = np.array([13,28, 11, 100])
plt.plot(y, linestyle = "dotted", color = "r")
plt.show()
#temelde 3 tane çizgi tipi var: dotted, linestyle, dashsot
#çizgiye renk özelliği vermek istiyorsan colar argümanını tanımla.

#Multiple Line
x=np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])
plt.plot(x)
plt.plot(y)
plt.show

#LABELS (Etiketler)
x=np.array([80, 85, 90, 95, 100])
y = np.array([240, 250, 260, 270, 280])
plt.plot(x,y)
plt.title("grafiğe ne başlık vermek istiyorsan onu yaz")
#x ekseninde isimlendirme yapmak istersek
plt.xlabel
#y ekseninde isimlendirme
plt.ylabel
#grafiğin arka tarafına ızgara koymak için
plt.grid()
#son olarak her zaman plt.show() diyerek görselleştir.
plt.show()

#subplots özelliği; birlikte birden fazla görselin gösterilmeye çalışması
#elimizde yer alan 2 görselin (plot1 ve plot2) görselleştirilmesini yapalım.
###plot 1
x=np.array([80, 85, 90, 95, 100])
y = np.array([240, 250, 260, 270, 280])
#plt.subplot(a satırlı, b sütunlu grafik oluştur, şuan bunun c.grafiğini oluşturuyorum)
plt.subplot(1,2,1)
plt.title("1")
plt.plot(x,y)
plt.show()

###plot2
x=np.array([80, 85, 90, 95, 100])
y = np.array([240, 250, 260, 270, 280])
plt.subplot(1,2,2)
plt.title("2")
plt.plot(x,y)
plt.show()

############# SEABORN İLE VERİ GÖRSELLEŞTİRME
#yüksek seviyeli veri görselleştirmek için kullanılan kütüphanedir. daha kısa zamanda çok iş yapar.
#seaborn ile kategorik değişkenleri görselleştirme
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df=sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
#countplot(verinin adı, ilgili veri setini gir)///// aynı zamanda hanigi eksende nerede oldugunu belirtlmelisin
sns.countplot(x = df["sex"], data= df)
plt.show()

#seaborn ile sayısal değişkenleri görselleştirme
sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].his()
plt.show()

#GELİŞMİŞ FONKSİYONEL KEŞİFCİ DEĞİŞKEN ANALİZİ
#amaç elimizdeki verileri fonksiyonel tarzda işleyebilmeyi, veriler hakkında hızlı bir içgörü elde etmeyi sağlar.
#hızlı bir şekilde genel fonksiyonlar ile elimizdeki verileri analiz etmek
#1.Genel Resim
#veri setinin dış ve iç özelliklerini genel haliyle bilmek. kaç gözlem, kaç değişken, değişken tiplerini vs var.
#2.Kategorik Değişken Analizi
#3.Sayısal Değişken Analizi
#4.Hedef Değişken Analizi
#5.Korelasyon Analizi

#1.Genel Resim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
ddf = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
#sayısal değişkenleri betimleme
df.describe().T
#eksik değer var mı yok mu?
df.isnull().values.any()
#veri setindeki eksik değer sayısını veren fonksiyon
df.isnull().sum()

#yaptiğimiz değişiklikler nasıl gözüküyor. bunun için fonskiyon yazalım
#BÖYLECE GENEL RESMİ GÖRMÜŞ OLACAĞIZ
def check_df(dataframe,  head=5):
    print("######SHAPE#######")
    print(dataframe.shape)
    #değişkenlerdeki tip bilgisini sorgulamak için
    print(dataframe.dtypes)
    print( "######SHAPE#######")
    print(dataframe.head(head))
    print( "######SHAPE#######")
    print(dataframe.tail(head))
    print( "######SHAPE#######" )
    print(fataframe.isnull().sum())
    print( "######SHAPE#######" )
    print(dataframe.describe([0,0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)

# yeni bir veri seti getirip, yukarıdaki fonksiyonla okutmak istiyorum diyelim.
check_df(df)
df = sns.load_dataset("tips")
check_df(df)

#KATEGORİK DEĞİŞKEN ANALİZİ
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
pd.set_option("display.max_columns",None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
#tek bir değişkeni analiz etmek istediğinde, hangi özelliği elde etmek istiyorsak onunla çağırırız.
df["embarked"].value_counts()
df["sex"].unique()
#toplamda kaçç tane eşşsiz sınıf var.
df["sex"].nunique()

#veri setinde çok fazla değişen olduğunda tek tek ele almam zor.
#diyelim ki veri seti içinden otomatikmen tüm kategorik değişkenleri seçsin istiyorum.
#bunu yaparken hem tip bilgisine göre yapacağız hem de tip bilgisi farklı olduğu halde kategorik olan değişkenleri yakalamamız lazım.
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category, object, bool"]]


#veri setindeki bazı değişkenler var ki onları yakalamak zor. kategorik olmasına rağmen bir önceki fonskiyonla bazı değişkenleri yakalayamayız.
#mesela "survived" değişekni 0-1 lerden oluştuğu için bunu yakalamak zor. bu ve bunun gibiler için farklı bir method geliştirmelisin:
#10' dan küçük eşsiz sınıf sayısına sahip olan VE tipleri int-float ise bu benim için "NUMERİK GÖRÜNÜMLÜ KATEGORİK DEĞİŞKENDİR" diye bir algoritma kuruyorum.
num_but_cat = [col for col in df.columns if df[col].nunique() <10 and df[col].dtypes in ["int","float"]]

#object ve category türde olup sınıf sayısı çok fazla olan değişkenler olabilir.
#bunlara "cardinelesi yüksek değişkenler" denir.
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

#cat_cols ve num_but_cat aynı şey o zaman ekle üzerine
cat_cols = cat_cols + num_but_cat

# eğer aynı şey olmasaydı o zaman çıkarma yapmalıydın.
cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols]

#seçtiğimiz değişkenlerin eşsiz sınıf saysına bakalım
df[cat_cols].nunique()

#cat_cols içerisinde olmayanlara bakalım.
[col for col in df.columns if col not in cat_cols]

#şimdi seçtiklerimize bir fonksiyon yazalım. Nasıl;
# fonskdsiyona girilen değerlerin value_counts() alsın. Hangi sınıftan kaçar tane var.
df["survived"].value_counts()
#sınıfların yüzdelik bilgisini yazdır.
100 * df["survived"].value_counts() /len(df)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100* dataframe[col_name].value_counts() / len(dataframe)}))
    print("#################")
#değişkenleri aratmak için
cat_summary(df,"sex")
#binlerce değişken arasından tek tek aratmak çok zor iş bu nedenle;
 for col in cat_cols:
     cat_summary(df, col)

#CAT_SUMMARY FONKSİYONUNA GRAFİK ÖZELLİĞİNİDE EKLEYELİM.
#plot ön tanımlı olarak false yaptım
def cat_summary(dataframe, col_name, plot = False):
    print( pd.DataFrame( {col_name: dataframe[col_name].value_counts(),
                          "Ratio": 100 * dataframe[col_name].value_counts() / len( dataframe )} ) )
print("#################")



#eğer plot açıksa;
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)
cat_summary(df, "sex", plot=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("veri görselleştirilmiyor")
    else:
          cat_summary(df, col, plot=True)



#adult_male verisi true-false oluşuyor. bunu astype ile 1-0 lara çevir.
df["adult_male"].astype(int)

#şimdi bu işlemi fonskiyonel olarak yapalım.
for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary( df, col, plot=True )
    else:
          cat_summary(df, col, plot=True)


#################Sayısal Değişken Analizi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplotas plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df= sns.load_dataset("titanic")
df.head()
#AGE VE FARE değişkenlerinin betimsel istatistiklerine ulaşmak istiyorum.
df[["age", "fare"]].describe().T
num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, numerical_col):
    quantiles=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    print(dataframe[numerical_col].describe(quantiles).T)

num_summary(df, "age")

#daha fazla sayıda değişken olduğunda böyle tek tek sorgulama yapmak zor o yüzden döngü yaz.
for col in num_cols:
    num_summary(df,col)

def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    print( dataframe[numerical_col].describe( quantiles ).T )

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block = True)
    num_summary(df, "age", plot=True)

    for col in num_cols:
         num_summary(df, col, plot= True)



######DEĞİŞKENLERİN OTOMATİKMEN YAKALANMASI VE İŞLEMLERİN GENELLEŞTİRİLMESİ
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplotas plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df= sns.load_dataset("titanic")
df.head()
df.info()

#bir değişkenin eşsiz sınıf sayısı 10 ve 10 dan azsa categorik değişkendir(cat_th)
#eğer 20 ve 20 den düşükse cardinal değişken muamelesi yapacağız(car_th)
#fonksiyona doküman yaz. docstring.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakar kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
    değişken isimleri alınmak istenen dataframe dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için eşik değeri verir.
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols:list
        kategorik değişkenlerin listesi
    num_cols: list
        numerik değişkenlerin bir listesi
    cat_but_car: list
        kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols un içeriisnde
    """
#yazmış olduğun docstringi çağır
    help(grab_col_names)

#cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category, object, bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() <10 and df[col].dtypes in ["int","float"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

#raporlama yap
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_col, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name):
    print( pd.DataFrame( {col_name: dataframe[col_name].value_counts(),
                          "Ratio": 100 * dataframe[col_name].value_counts() } ) )
print("#################")
cat_summary((df,  "sex"))

for col in cat_cols:
    cat_summary(df,col)


def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    print( dataframe[numerical_col].describe( quantiles ).T )

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block = True)

for col in num_cols:
    num_summary(df,col, plot=True)


#######veri setini oku, tipi değiştir ve görselleştir.
df = sns.load_dataset("titanic")
df.info()
for col in df.columns:
    if df[col].dtype == "bool":
        df[col] = df[col].astype(int)

df.info()
#grab fonskiyonunu çağır
cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot = False):
    print( pd.DataFrame( {col_name: dataframe[col_name].value_counts(),
                          "Ratio": 100 * dataframe[col_name].value_counts() / len( dataframe )} ) )
    print("#################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df,col, plot= True)

for col in num_cols:
    num_summary(df, col, plot=True)


#####HEDEF DEĞİŞKEN ANALİZİ
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

def cat_summary(dataframe, col_name, plot = False):
    print( pd.DataFrame( {col_name: dataframe[col_name].value_counts(),
                          "Ratio": 100 * dataframe[col_name].value_counts() / len( dataframe )} ) )
    print("#################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df,col, plot= True)

for col in num_cols:
    num_summary(df, col, plot=True)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category, object, bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() <10 and df[col].dtypes in ["int","float"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]


    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_col, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
#ilgili hedef değişkeni analiz etmeye geldi sıra. burada hedef değişkenimiz survived()
df["survived"].value_counts()
#ya da
cat_summary(df, "survived")



##HEDEF DEĞİŞKENİN KATEGORİK DEĞİŞKENLER İLE ANALİZİ
df.groupby("sex")["survived"].mean()

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN" : dataframe.groupby(categorical_col)[target].mean()}))

#pclass a göre survived durumunu incele
target_summary_with_cat(df, "survived", "pclass")

#her değişkeni böyle incelemek zaman alır o yüzden döngü yaz
for col in cat_cols:
    target_summary_with_cat(df,"survived", col)


#HEDEF DEĞİŞKENİN SAYISAL DEĞŞKENLER İLE ANALİZİ
#Bu seferde groupby kısmına bağımlı değişkeni, agg ksımına ise bağımsız değişkeni gönder.
df.groupby("survived")["age"].mean()
#ya da
df.groupby("survived").agg({"age": "mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end= "\n\n\n")

target_summary_with_num(df, "survived", "age")

for col in num_cols:
    target_summary_with_num(df,"survived", col)


##KOREALASYON ANALİZİ
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv("datasets/breast_cancer.csv")
#gereksiz değişkenlerden kurtulmak istediğimiz için 1 den -1 e kadar git dedik.
df = df.iloc[:, 1:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtype in [int,float]]
corr = df[num_cols].corr()
#şimdi bir ısı haritası oluşturalım.
#oluşturacağımız grafik 12-12 lik olsun istiyorum
sns.set(rc={"figure.figsize": (12,12)})
sns.heatmap(corr, cmap = "RdBu")
plt.show()

#yüksek korelasyonlu değişkenlerin silinmesi
cor_matrix = df.corr().abs()
#bu matris de 0-1 in ilişkisi ile 1-0 ın ilişkisi aynı şey ama tabloda 2 yer kaplıyor boş yere.
#işte bu gereksizlerden kurtulmak lazım.

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90) ]
#yüksek koreasyonlu olanları seçmek için
cor_matrix[drop_list]
#yüksek korealsyonlu değerleri silmek için
df.drop(drop_list, axis=1)

def (high_correlated_cols(dataframe, plot=False, corr_th=0.9):
    corr=dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize": (15,15)})
        sns.heatmap(corr, cmap= "RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot = True)
df.drop(drop_list, axis=1)

high_correlated_cols(df.drop(drop_list, axis=1), plot =True)

##yaklaşık 600 mb lık 300 den fazla değişkenin olduğu bir veri setinde deneyelim.
##kaggle- "train_transection.csv"

df=pd.read_csv("datasets/freud_train_transection.csv")
len(df.columns)
df.head()
drop_list=high_correlated_cols(df, plot=True)

len(df.drop(drop_list, axis=1).columns)

