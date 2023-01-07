#GÖREV1
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option("display.max_columns",None)

df=sns.load_dataset("car_crashes")
df.columns
df.info()

["NUM_" + col.upper() if df[col].dtype != 0 else col.upper() for col in df.columns]

#GÖREV2
[col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]

#görev3
og_list = ["abbrev", "no_previous"]
new_cols = [col for col in df.columns if col not in og_list]
new_df=df[new_cols]
new_df.head()

################# PANDAS ALIŞTIRMALAR
#Görev1
import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

df=sns.load_dataset("titanic")
df.head()
df.shape

#GÖREV2
df["sex"].value_counts()

#görev3
df.nunique()

#görev4
df["pclass"].unique()

#görev5
df[["pclass", "parch"]].nunique()

#görev6
df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtype
df.info()

#görev7
df[df["embarked"] == "C"].head(5)

#görev8
df[df["embarked"] != "S"].head(5)
df[df["embarked"] != "S"]["embarked"].unique()
#veya
df[~(df["embarked"] == "S")]["embarked"].unique()

#GÖREV9
df[(df["age"]<30) & (df["sex"] == "female")].head()

#GÖREV10
df[(df["fare"] > 500) | (df["age"] > 70)].head()

#GÖREV11 (eksik değer analizi)
df.isnull().sum()

#GÖREV12
df.drop("who", axis=1, inplace=True)

#GÖREV13
type(df["deck"].mode())
df["deck"].mode()[0]
df["deck"].fillna(df["deck"].mode()[0], inplace=True)
df["deck"].isnull().sum()

#GÖREV14
df["age"].fillna(df["age"].median(),inplace=True)
df.isnull().sum()

#GÖREV15
df.groupby(["pclass","sex"]).agg({"survived": ["sum","count","mean"]})

#GÖREV16
def age_30(age):
    if age < 30:
        return 1
    else:
        return 0

df["age_flag"] = df["age"].apply(lambda x : age_30(x))

df["age_flag"] = df["age"].apply(lambda x: 1 if x<30 else 0)

#GÖREV17
df = sns.load_dataset("tips")
df.head()
df.shape

#GÖREV18
df.groupby("time").agg({"total_bill": ["sum","min","mean","max"]})

#GÖREV19
df.groupby(["day","time"]).agg({"total_bill": ["sum","min","mean","max"]})

#GÖREV20
df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill": ["sum","min","max","mean"],
                                                                           "tip":  ["sum","min","max","mean"],
                                                                            "Lunch" : lambda x:  x.nunqiue()})

#GÖREV21
df.loc[(df["size"] < 3) & (df["total_bill"] >10 ) , "total_bill"].mean() # 17.184965034965035

#GÖREV22
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()

#GÖREV23
new_df = df.sort_values("total_bill_tip_sum", ascending=False)[:30]
new_df.shape



