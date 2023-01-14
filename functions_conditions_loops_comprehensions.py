# Fonksiyonları biçimlendirmek için ?print gibi yaz ve neleri kullanabileceğini gör
# sep() ; iki string arasına boşluk konulmasını istediğimizde veya ne istersen
print("a", "b")
print("a", "b", sep="__")


# fonksiyon tanımlama
def calculate(x):
    print(x * 2)

calculate(5)


# 2parametreli fonksiyon nasıl yazılır
def summer(arg1, arg2):
    print(arg1 + arg2)


summer(7, 8)
summer(arg1=7, arg2=8)


# docstring
def summer(arg1, arg2):
    """
    Sum of two numbers

    Parameters
    arg1: int, float
        #buraya arg1 hakkında bilgi verebilirsin
    arg2: int, float

    Returns
        int, float
    """
    print(arg1 + arg2)


summer(1, 3)


# yukarıda summer fonksiyonunun docstrıngını yazdık, summerin üstüne gelince nasıl bir fonksiyon oldugunun bilgilerini bize verecektir.
# summer fonks. sende erişmek istiyorsan console kısmına ?summer veya help(summer) yaz.


# Fonksiyonlarda STATEMENT/BODY BÖLÜMÜ
# def function_name(parametres-arguments):
# statements (function body)

def say_hi():
    print("Merhaba")
    print("hi")
    print("hello")


say_hi()


def say_hi(string):
    print(string)


say_hi("miuul")


def multiplication(a, b):
    c = a * b
    print(c)


multiplication(10, 9)

# girilen değerleri bir liste içinde saklayacak bir fonksiyon tanımlayalım.
list_store = []


def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


add_element(3, 4)
add_element(16, 8)


# Ön tanımlı argümanlar/parametreler (default parametres/arguments)
def divide(a, b):
    print(a / b)


divide(1, 2)


# bazı arg için tanımlı değer ekleyerek kullanıcılar her bir arg girmese dahi çalışmasını sağlarız
def divide(a, b=1):
    print(a / b)


divide(1)


# kullanıcı stringe hiçbir şey girmedi o zaman sen merhabe de :)
def say_hi(string="MERHABA"):
    print(string)


say_hi()
say_hi("mrb")


# ne zaman fonksiyon yazma ihtiyacımız olur?
# diyelim ki akıllı sokak lambaları yapacaksın neyi bilmen lazım? ısı sıcaklık pil durumunu
# varm, moisture, charge
# Dont Repait Yourself (DRY): Bilgiler kendini tekrar ettiğinde otur fonksıyon yaz

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)


calculate(98, 12, 87)


# Return: Fonksıyon çıktılarını girdi olarak kullanılan fonksiyondur.
def calculate(varm, moisture, charge):
    return (varm + moisture) / charge


calculate(98, 12, 87) * 10
a = calculate(98, 12, 87) * 10
print(a)


def calculate(varm, moisture, charge):
    varm = varm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (varm + moisture) / charge
    return varm, moisture, charge, output


calculate(98, 12, 87)
type(calculate(98, 12, 87))


# fonkdiyon içerisnde fonksiyon çağırmak

def calculate(varm, moisture, charge):
    return int((varm + moisture) / charge)


calculate(90, 12, 12) * 10


def standardization(a, p):
    return a * 10 / 100 * p * p


standardization(45, 1)
def all_calculation(varm, moisture, charge, p):
    a = calculate(varm, moisture, charge)
    b = standardization(a, p)
    print(b * 10)
all_calculation(1, 3, 5, 12)

#############################################3

def all_calculation(varm, moisture, charge,a, p):
    print(calculate(varm, moisture, charge))
    b = standardization(a, p)
    print(b * 10)
all_calculation(1, 3, 5, 19, 12)

#KOŞULLAR (CONDİTTONS)

#İF
if 1 == 1:
    print("something")

###

def number_check(number):
    if number == 10:
        print("number is 10")
number_check(12)


###
def number_check(number):
    if number == 10:
        print("number is 10")
number_check(10)

###
def number_check(number):
    if number == 10:
        print("number is 10")
    else:
        print("number is not 10")
number_check(12)


#ELİF
def number_check(number):
    if number > 10:
        print("greater than 10")
    elif number < 10:
        print("less than 10 ")
    else:
        print("equal to 10")
number_check(14)

#DÖNGÜLER (LOOPS)
students = ["John", "Mark", "Venessa", "Mariam"]
students[3]

for student in students:
    print(student)
for student in students:
    print(student.upper())

salaries = [1000, 2000, 3000, 4000, 5000]
for salary in salaries:
    print (salary)

for salary in salaries :
    print(int(salary*20/100)+salary)

#DRY . BU NEDENLE BİR FONKSİYON TANIMLA
def new_salary(salary, rate):
    return int(salary*rate/100 +salary)
new_salary(1500,10)

#bütün maaşlara %10zam yaptığını düşün
for salary in salaries:
    print(new_salary(salary,10))

#maaşların miktarına göre farklı zam yapıcam
for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary,10))
    else:
        print(new_salary(salary,20))

#Aşağıda verilen stringlerin index numaraları tekse küçült, çiftse büyüt.
#before: "hi my name is John"
#after: "Hi mY NaMe iS JoHn"

#len(), indexlerin uzunlupunu verir.
#range(), iki dğer arasında sayı üretme imkanı sağlar.

range(len("miull"))
range(0,5)
for i in range(len("miuul")):
    print(i)

def alternating(string):
     new_string = ""
     for string_index in range(len(string)):
         if string_index %2 == 0 :
             new_string += string[string_index].upper()
     else:
         new_string += string[string_index].lower()

    print(new_string)
alternating("miull")

#BREAK: Aranan koşul yakalandığında döngünün durmasını sağlar.
salaries = [1000, 2000, 3000, 4000, 5000]
for salary in salaries:
    if salary ==3000:
    break
    print(salary)

#CONTİNUE: Aranan koşula geldiğinde bırak devam et sen işine.
for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

#WHİLE: -dı sürece.
number =1
while number<5:
    print(number)
    number +=1

#ENUMERATE: otomatik counter/ indexer ile for loop
#diyelim ki bir liste içerisindeki elemanlara ulaşıp belirli bir işlem uygulayacaksın
# aynı zamanda işlemin uygulanmış olduğu elemanların index bilgisini tutmaya sağlar.
#böylece tutmuş olduğu bu indexle de farklı işlemler yürütebiliriz.

students = ["John", "Mark", "Venessa", "Mariam"]
for student in students:
    print (student)
for index, student in enumerate(students):
    print(index, student)

#tek indexdekileri bir listede, çift indexdekileri farklı bir listede tut
A= []
B= []
for index, student in enumerate(students):
    if index %2 ==0:
        A.append(student)
    else:
        B.append(student)
    print(index, student)
A = []
B = []

# divide_students fonksiyonu yaz.
#çift indexde yer alan öğrenciler bir listede,
#tek indexde yer alan öğrenciler başka listede tut,
#fakat bu iki liste tek bir liste olarak return olsun.

students = ["John", "Mark", "Venessa", "Mariam"]
def divide_students (students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index %2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
        print(groups)
        return groups
divide_students(student)
st= divide_students(student)
st[0]
st[1]

#alternating fonksiyonunu enumerate ile yazılması
#Aşağıda verilen stringlerin index numaraları tekse küçült, çiftse büyüt.
#enumerate yerine range ve len fonksiyonları kullanılabilir ancak okunabilirlik anlaşılabilirlik için enumerate kullan.
def alternating_with_enumerate(string):
    new_string = ""
    for i , letter in enumerate(string):
        if i%2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)
alternating_with_enumerate("hi my name is john")

#zip: Ayrı listeleri tek bir liste içeriisnde her birinde bulunan elemanların sırasına uygun olarak birleştirir.

students = ["John", "Mark", "Venessa"]
departments = [ "math", "science", "physics"]
ages = [23, 45, 32]

list(zip(students, departments, ages))

#lambda: fonksiyon tanımlama şeklidir. kullan at fonksiyonlarından biridir.. yani bir tanımlama atama işlemi yapılmaksızın kullanılabilirdir.
#map: ilgili fonksiyonu listeye uygular
#filter:
#reduce: indirgemek

def summer(a,b):
    return a+b
summer(1,3)*9

new_sum = lambda a, b: a+b
new_sum(4, 5)

#map
salaries = [1000, 2000,3000, 4000]
###########lambda, map, filter, reduce pek anlamadım.


#LİST COMPREHENSİON
salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x*20 / 100 + x

for salary in salaries:
    print(new_salary(salary))

null_list = []
for salary in salaries:
    null_list.append(new_salary(salary))
print(null_list)

null_list = []
for salary in salaries:
    if salary>3000:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary*2))
print(null_list)

#list comprehensions yapısıyla yazalım
# list comprehension oluşturmak klasik liste oluşturarak başlanır;
[salary*2 for salary in salaries ]
#eğer tek başına if kullanacaksan bu en sağda olur
[salary*2 for salary in salaries if salary < 3000 ]
#eğer if else varsa o zman for en sağda olur
[salary*2 if salary < 3000  else salary*0 for salary in salaries ]

#list comprehinsionsla bir satırda birçok işlev yürütebilirim.
[new_salary(salary*2) if salary < 3000  else new_salary(salary*0,2)for salary in salaries ]

#student_no dakiler küçük harfle, diğerleri büyük harfle yazılsın
students = ["John", "Mark", "Venessa", "Mariam"]
students_no = ["John", "Venessa"]

[student.lower() if student in students_no else student.upper() for student in students]

#DİCT COMPREHENSİON
dictionary = {"a": 1,
              "b": 2,
              "c": 3,
              "d": 4}
dictionary.items()
{k: v**2 for (k,v) in dictionary.items()}

#böylece key ve value lara özel bir şekilde müdahale edebiliriz.
{k.upper(): v**2 for(k,v) in dictionary.items()}

#listedeki çift sayıların karesini alarak bir sözlüğe ekle.
#Key' ler orijinal değerler value' lar ise değiştirilmiş değerler olacak
numbers = range(10)
new_dict = {}
for n in numbers:
    if n %2 == 0:
        #key lere dokunma ama value ları 2 ile çarp
        new_dict[n] = n**2

#veya dict comprehensions ile yaz. dikkat et k sabıt, value karesını alıyor.
{n: n**2 for n in numbers if n %2 == 0}

#LİST & DİCT COMPREHENSİON UYGULAMALARI
#bir veri setindeki değişken isimlerini değiştirmek

#bir kütüphaneyi (örneğin seaborn) import etmek için; import seaborn as sns
#.columns: ilgili data frame değişkenlerininin ismi gelir.
import seaborn as sns
df = sns.load_dataset("car_crashes" )
df.columns

A=[]
for col in df.columns:
    A.append(col.upper())

df.columns = A

#Aynı işlemi COMPREHENSİONS ile yap
df.columns = A
df= sns.load_dataset("car_crashes")
df.columns = [col.upper() for col in df.columns]

#isminde "INS" olan değişkenlerin başına "FLAG" diğerlerine "NO_FLAG" ekle
#içeriisnde INS olanları getir
[col for col in df.columns if "INS" in col]

#INS olanların başına "FLAG" ekle
["FLAG_" +col for col in df.columns if "INS" in col]

#Aynı zamansa ins yoksa NON_FLAG_ ekle
["FLAG_" + col if "INS" in col else "NO_FLAG_" +col for col in df.columns]

#değiştirilen değişken isimlerini kalıcı olarak data frame adı olarak tutmak istediğinde
df.columns = ["FLAG_" + col if "INS" in col else "NO_FLAG_" +col for col in df.columns]
#########################################################
#Amaç KEY İ string, Value si aşağıdaki, gibi bir liste olan sözlük oluşturma
#işlemi sadeece sayısal değişkenler için yapmak istiyoruz.
import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns
#"O" object demektir. yani string tipindeki değişkenlerdir.
num_cols = [col for col in df. columns if df[col].dtype != "O"]
dict = {}
agg_list = ["mean", "min", "max", "sum"]

for col in num_cols:
    dict[col] = agg_list
print(dict)

#comprehensionsları kullanarak kısa yoldan yapalım
new_dict = {col: agg_list for col in num_cols}

df[num_cols].head()
df[num_cols].agg(new_dict)

wages = [1000,2000,3000]
new_wages = lambda x: x*0.2 + x
list(map(new_wages, wages))
