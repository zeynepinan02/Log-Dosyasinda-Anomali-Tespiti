from sklearn.model_selection import train_test_split
from io import StringIO
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

#log_file_path = 'C:/Users/zeyze/Desktop/tasarım-proje/log-girdi/FGT60E4Q16043370db.log20230804.json'
#C:/Users/zeyze/Desktop/tasarım-proje/log-girdi/FGT60E4Q16043370db.log20230804.json

#dosyayı kullanicidan alır.
log_file_path =input("json dosyasının yolunu giriniz: ")

#dosyadan verileri okuyor
with open(log_file_path, 'r') as file:
    json_data = file.read()

#okunan verileri data frame oluşturup içine koyuyor.
#eski sürümlerde read_json fonksiyonu direkt dosyayı okuyabiliyordu artık stringIO tipini parametre alıyor.
df = pd.read_json(StringIO(json_data))

#kalmasını istedigim sutunlar(time ı cikardim unutma)
columns = ['action','app','appcat','datetime','devtype','dstcountry','dstintf','dstip','dstosname','dstport','duration','eventtime','msg','osname','policyid','policyname','policytype','rcvdbyte','sentbyte','service','srccountry','srcintf','srcip','srcmac','srcname','srcport','srcserver','subtype']

#baska bir data frame olusturup icine bu sutunları verileriyle beraber atıyoruz. içinde bu sutunların bulunmadıgı verilere sutunları ekler degerleri null yapar.
df_newColums= df[columns].copy()

filter_column = 'subtype'
filter_value= ['forward','local','system','wireless','sdwan','app-ctrl','webfilter','ssl']

#belirledigimiz kolonun degerlerine gore filtreleme yapıyor. Kosulu saglayan satırları data frame'e kaydediyoruz.
df_newColums=df_newColums[df_newColums[filter_column].isin(filter_value)]

#json veri setinde dateTime alanı $date seklinde etiketli halde bulunuyor. Bunu dateTime tipine donusturur.
df_newColums["datetime"] = pd.to_datetime(df_newColums["datetime"].apply(lambda x: x["$date"]))

# veri setinde örnekleme yapıyor

# Sınıflara göre gruplandırın
logs = df_newColums.groupby('datetime')

#birden fazla sütunla gruplama yapmak istersek
#logs = df_newColums.groupby(['date', 'time'])

ornekler = []

# Her sınıftan eşit sayıda örnek alın
for log_name, log in logs:
    # Belirlediğiniz örnek sayısını ayarlayın (örneğin 50)
    ornek_grup = log.sample(n=1, random_state=42)
    ornekler.append(ornek_grup)

# Stratified örnekleri birleştirin
stratified_ornekler = pd.concat(ornekler)

#datetime sutununu tekrar jsona uygun hale getiriyor
stratified_ornekler["datetime"] = stratified_ornekler["datetime"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))

# Sonuçları yazdırın
stratified_ornekler.to_json('DataSet.json', orient='records')


