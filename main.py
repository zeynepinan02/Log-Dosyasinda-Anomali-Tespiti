from sklearn.model_selection import train_test_split
from io import StringIO
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import tensorflow as tf
from keras.layers import Input, LSTM, RepeatVector, Dense, TimeDistributed
from keras.models import Model
from keras import regularizers


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



'''

#kalmasını istedigim sutunlar(time ı cikardim unutma)
columns = ['action','app','appcat','datetime','devtype','dstcountry','dstintf','dstip','dstport','duration',
           'policyid','policytype','rcvdbyte','sentbyte','service','srccountry','srcintf','srcip','srcmac',
           'subtype']

#baska bir data frame olusturup icine bu sutunları verileriyle beraber atıyoruz. içinde bu sutunların bulunmadıgı verilere sutunları ekler degerleri null yapar.
df_newColums= df[columns].copy()

#df ile işimiz bitti
del df

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

#df_newColums ile işimiz bitti
del df_newColums

#datetime sutununu tekrar jsona uygun hale getiriyor
#stratified_ornekler["datetime"] = stratified_ornekler["datetime"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))




'''


"""
dict_mask = stratified_ornekler["eventtime"].apply(lambda x: isinstance(x, dict))

# Dict tipindeki verilerin sayısını bulun
dict_count = dict_mask.sum()

#Eğer en az bir dict tipinde değer varsa, hataya neden olan değeri bulun
if dict_count > 0:
    error_value = stratified_ornekler["eventtime"][dict_mask].iloc[0]
    print(f"Sütundaki Dict Tipindeki Veri Sayısı: {dict_count}")
    print(f"Hata Yapan Değer: {error_value}")
else:
    print("Sütunda Dict Tipinde Veri Bulunmuyor.")
"""






'''
#sayısallaştırma yapabilmek için sütunları düzenleyelim

#dict tipindeki  verileri null yapıyor.
for column_name in stratified_ornekler.columns:
    if column_name != "datetime":
        stratified_ornekler[column_name] = stratified_ornekler[column_name].apply(lambda x: None if isinstance(x, dict) else x)

# app sütununu düzenler
stratified_ornekler['app'] = stratified_ornekler['app'].apply(lambda x: str(x).split('/')[0] if pd.notna(x) else x)

#FrequencyEncoded ile sayısallaştırma işlemi
for objectColumn in stratified_ornekler.select_dtypes(include='object').columns:
    if objectColumn != "datetime" and objectColumn:
        #frekansları hesaplar
        category_freq = stratified_ornekler[objectColumn].value_counts(normalize=True)
        #frekans degerlerine gore sütunu düzenler
        stratified_ornekler[objectColumn] = stratified_ornekler[objectColumn].map(category_freq)

#null deger çok ise ve kategorik sütunsa ort, kategorik sütnlarda genellikle mod en çok tekrar edenle diğerleri arasında uçurum varsa medyan, int değerlerde kesinlikle ort değil aykırı değerler etkilemesin diye
medyan_columns = ['appcat','dstport','duration']
mod_columns = ['action','dstcountry','dstintf','dstip','policyid','rcvdbyte','sentbyte','service','srcintf','srcip',
               'subtype']
ort_columns = ['app','devtype','policytype','srccountry','srcmac']
int_columns = ['dstport','duration','policyid','rcvdbyte','sentbyte']


#ortalama,mod,medyan ile eksik veri tamamlama, seçili sütunlara normalizasyon uygulama
for column in stratified_ornekler.columns:
    if column in ort_columns:
        mean_value = stratified_ornekler[column].mean(skipna=True)
        stratified_ornekler[column].fillna(mean_value, inplace=True)
    if column in mod_columns:
        modes = mode(stratified_ornekler[column], nan_policy='omit')  # mode fonksiyonu ile mod değeri bulunuyor
        mode_value = modes.mode.max()
        stratified_ornekler[column].fillna(mode_value, inplace=True)
    if column in medyan_columns:
        medyan_value = np.nanmedian(stratified_ornekler[column])  # nanmedian fonksiyonu ile medyan değeri bulunuyor
        stratified_ornekler[column].fillna(medyan_value, inplace=True)
    if column in int_columns:
        scaler = MinMaxScaler()
        stratified_ornekler[column] = scaler.fit_transform(stratified_ornekler[column].values.reshape(-1, 1))

        #print(column)
        #print(stratified_ornekler[column].max())
        #print(stratified_ornekler[column].min())
'''







"""
#pca nın gerekliliğine ve boyutuna karar vermek için datetime içermeyen veri seti oluşturduk
columns = ['action','app','appcat','devtype','dstcountry','dstintf','dstip','dstport','duration',
           'policyid','policytype','rcvdbyte','sentbyte','service','srccountry','srcintf','srcip','srcmac',
           'subtype']
pcaAnalyzeMatris = stratified_ornekler[columns].copy()
"""

"""
#pca nın gerekliliğine karar vermek için şartları kontrol ediyoruz

#VIF degerini kontrol ediyoruz VIF>5 ise pca uygulanması mantıklı
vif_data = pd.DataFrame()
vif_data["Variable"] = pcaAnalyzeMatris.columns
vif_data["VIF"] = [variance_inflation_factor(pcaAnalyzeMatris.values, i) for i in range(pcaAnalyzeMatris.shape[1])]
print(vif_data)

#korelasyon matrisi oluşturuyoruz.
correlation_matrix = pcaAnalyzeMatris.corr()
print(correlation_matrix)

#koveryans matrisi oluşturuyoruz. Düşük varyans zayıf ilişki pca uygulanabilir.
cov_matrix = np.cov(pcaAnalyzeMatris, rowvar=False)  # rowvar=False, değişkenler sütunlarda
cov_df = pd.DataFrame(cov_matrix, columns=pcaAnalyzeMatris.columns, index=pcaAnalyzeMatris.columns)

#Matrisi ekrana yazdır
print("Kovaryans Matrisi:")
print(cov_df)
 """

""" 
#yamaç eğrisi grafiği ile pca nın boyutuna karar veriyoruz.

#pca modelini oluşturuyoruz
pca = PCA()
pca.fit(pcaAnalyzeMatris)

# Açıklanan varyans oranlarını al
explained_variance_ratio = pca.explained_variance_ratio_

# Toplam açıklanan varyans oranını hesapla
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Açıklanan varyans oranlarını ve kümülatif varyans oranlarını çiz
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', label='Varyans Oranı')
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', label='Kümülatif Varyans Oranı')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('Varyans Oranı')
plt.title('PCA - Açıklanan Varyans Oranları')
plt.legend()
plt.show()
"""






'''
# Tarih/saat sütununu string'ten datetime'a dönüştürme
#stratified_ornekler['datetime'] = pd.to_datetime(stratified_ornekler['datetime'],format='%Y.%m.%d.%H.%M.%S')

# datetime sütununu sakla
hidden_column = stratified_ornekler['datetime']
stratified_ornekler.drop('datetime', axis=1, inplace=True)


#pca yı yapıyoruz

# PCA modelini oluştur (svd_solver='full' yani korelasyon matrisini kullanarak)
pca = PCA(n_components=4)
pca_result = pca.fit_transform(stratified_ornekler)

#stratified_ornekler ile işimiz bitti
del stratified_ornekler

# PCA sonuçlarını yeni bir DataFrame'e ekle
result_df = pd.DataFrame(data=pca_result, columns=[f'PC{i}' for i in range(1, 5)])
#result_df = pd.DataFrame(pca_result, columns=[f'pca_{i}' for i in range(pca_result.shape[1])])


#result_df.to_json('Result_df.json', orient='records')

# datetime sütununu tekrar ekleyerek sonuçları birleştir
pca_df = pd.concat([hidden_column.reset_index(drop=True), result_df], axis=1)

#pca_df = pd.merge(hidden_column.to_frame(), result_df, left_on=hidden_column.index, right_index=True, how='left')

#pca_df.to_json('Merge_df.json', orient='records', date_format='iso')

merged_data = pd.DataFrame()

# Set 'datetime_column' as index
pca_df = pca_df.set_index('datetime')
merged_data = pd.concat([merged_data, pca_df], axis=1)
'''






"""
# Açıklanan varyans oranlarını al
explained_variance_ratio = pca.explained_variance_ratio_

# Her bir bileşenin açıklanan varyans oranını ekrana yazdır
for i, ratio in enumerate(explained_variance_ratio, 1):
    print(f'Bileşen {i}: {ratio:.4f}')
    
# Toplam varyansı bulun
total_variance = np.sum(explained_variance_ratio)
print(f"Toplam varyans: {total_variance * 100:.2f}%")
print(f'Toplam Açıklanan Varyans: {total_variance:.4f}')

# Sonuçları görselleştirin
plt.subplot(1, 3, 3)
plt.bar([f'PC{i}' for i in range(1, len(pca.explained_variance_ratio_) + 1)],
        pca.explained_variance_ratio_)
plt.title('Explained Variance with StandardScaler')
plt.ylabel('Explained Variance Ratio')
plt.show()

# İlk iki bileşeni görselleştir
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title('PCA: Korelasyon Matrisi ile')
plt.xlabel('Birinci Bileşen')
plt.ylabel('İkinci Bileşen')
plt.show()
print("PCA Sonuçları:")
print(pca_result)
"""

"""
#qqplot grafiği çizer
sm.qqplot(stratified_ornekler['sentbyte'], line='45', fit=True)
plt.title('sentbyte QQ Plot')
plt.show()
"""
"""
#histogram grafiği çizer
plt.figure(figsize=(10, 6))
sns.histplot(x='sentbyte', data=stratified_ornekler, bins=7, kde=False, color='blue')
plt.title('sentbyte Histogram')
plt.xlabel('sentbyte')
plt.ylabel('Frequency')
plt.show()
"""

"""
#dict tipindeki verilerin değerini gösterir
for index, value in stratified_ornekler['datetime'].items():
    if isinstance(value, dict):
        print(f"Index: {index}, Değer: {value}")
"""






'''
# Veriyi eğitim ve test olarak bölme
split_ratio = 0.7  # Eğitim verisinin oranı, bu oranı ihtiyacınıza göre ayarlayabilirsiniz
split_index = int(split_ratio * len(merged_data))  # Verinin kaçıncı indeksine kadar eğitim verisi alınacak hesaplanıyor

train_data = merged_data[:split_index]   # Eğitim verisi, verinin başından belirlenen indekse kadar olan kısmı
test_data = merged_data[split_index:]   # Test verisi, belirlenen indeksten verinin sonuna kadar olan kısmı
'''







'''
split_index = 100  # İlk 10 veriyi eğitim verisi olarak kullan

train_data = pca_df[:split_index]   # Eğitim verisi, verinin başından belirlenen indekse kadar olan kısmı
test_data = pca_df[split_index:]    # Test verisi, verinin belirlenen indeksten sonraki kısmı
'''






'''
# Eğitim verisi
print("Eğitim Verisi:")
print(train_data.head())

# Test verisi
print("Test Verisi:")
print(test_data.head())
'''





'''
pca_df.to_json('DataSet.json', orient='records')
train_data.to_json('trainData.json', orient='records')
test_data.to_json('testData.json', orient='records')
'''

'''
#eğitim verilerinin zaman içindeki değişimini göteren grafik

fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(train_data['PC1'], label='PC1', color='blue', linewidth=1)
ax.plot(train_data['PC2'], label='PC2', color='red', linewidth=1)
ax.plot(train_data['PC3'], label='PC3', color='green', linewidth=1)
ax.plot(train_data['PC4'], label='PC4', color='black', linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Training Data', fontsize=16)
plt.show()


#test verilerinin zaman içindeki değişimini göteren grafik
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(test_data['PC1'], label='PC1', color='blue', linewidth=1)
ax.plot(test_data['PC2'], label='PC2', color='red', linewidth=1)
ax.plot(test_data['PC3'], label='PC3', color='green', linewidth=1)
ax.plot(test_data['PC4'], label='PC4', color='black', linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Test Data', fontsize=16)
plt.show()'''









'''
#numpy dizisine dönüştürür
train_data_np = train_data.values
test_data_np = test_data.values

print(train_data_np.shape)
print(test_data_np.shape)


# LSTM için giriş verilerini yeniden boyutlandırıyoruz [samples, timesteps, features]
train_data_reshaped = train_data_np.reshape(train_data_np.shape[0], 1, train_data_np.shape[1])#600 sn yani 10 dk lık time steps
print("Training data shape:", train_data_reshaped.shape)
test_data_reshaped = test_data_np.reshape(test_data_np.shape[0], 1, test_data_np.shape[1])
print("Test data shape:", test_data_reshaped.shape)


# autoencoder ağ modeli

def autoencoder_model(X):
    # Giriş katmanını tanımla
    inputs = Input(shape=(X.shape[1], X.shape[2]))

    #relu aktivasyon fonksiyonu bir alternatif olabilir
    # İlk LSTM katmanı: 16 birim, aktivasyon fonksiyonu 'relu', zaman dizisini döndür
    L1 = LSTM(16, activation='tanh', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
    #kernel_regularizer = regularizers.l1(0.00) bu da alternatif denersin 0.00 değerleriyle de oyna

    # İkinci LSTM katmanı: 4 birim, aktivasyon fonksiyonu 'relu', zaman dizisini döndürme
    L2 = LSTM(4, activation='tanh', return_sequences=False)(L1)

    #zaman adımındaki örüntüler için. Burada tekrarlama sayısını mı 10 dk periyot olarak ayarlıycaz yoksa zaman adımını mı
    # Giriş dizisini tekrar etme katmanı (RepeatVector)
    L3 = RepeatVector(X.shape[1])(L2)

    # Üçüncü LSTM katmanı: 4 birim, aktivasyon fonksiyonu 'relu', zaman dizisini döndür
    L4 = LSTM(4, activation='tanh', return_sequences=True)(L3)

    # Dördüncü LSTM katmanı: 16 birim, aktivasyon fonksiyonu 'relu', zaman dizisini döndür
    L5 = LSTM(16, activation='tanh', return_sequences=True)(L4)

    # Çıkış katmanını tanımla: Zamanla dağıtılmış (TimeDistributed) bir yoğun (Dense) katman
    output = TimeDistributed(Dense(X.shape[2], activation='tanh'))(L5)

    # Modeli oluştur
    model = Model(inputs=inputs, outputs=output)
    return model


#autoencoder model oluştur
model = autoencoder_model(train_data_reshaped)
#optimizer için alternatifler:adam,sgd,rmsprop #loss için alternatifler: mae,mse,huber
model.compile(optimizer='adam', loss='mae') #optimizer için alternatifler:adam,sgd,rmsprop #loss için alternatifler: mae,mse,huber
model.summary()

nb_epochs = 100
#her bir eğitim iterasyonunda kullanılacak örnek sayısını belirleyen bir parametre
batch_size = 10
#validation_split eğitim sırasında ayrılmış bir doğrulama seti kullanılmasını sağlar
history = model.fit(train_data_reshaped, train_data_reshaped, epochs=nb_epochs, batch_size=batch_size,validation_split=0.1).history
'''







'''
#eğitim hata grafiği oluşturur. Doğrulama örnekleri ve eğitim örnekleri için
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae) ')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()'''

#eğitim kayıp grafiği oluşturur
'''X_pred = model.predict(train_data_reshaped)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred_df = pd.DataFrame(X_pred, columns=train_data.columns)
X_pred_df.index = train_data.index
scored = pd.DataFrame(index=train_data. index)
Xtrain = train_data_reshaped.reshape(train_data_reshaped.shape[0], train_data_reshaped.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred_df-Xtrain), axis=1)
plt.figure(figsize=(16,9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.histplot(scored['Loss_mae'], bins=20, kde=True, color='blue')
plt.xlim([0, 0.2])
plt.show()'''











'''
# calculate the loss on the test set
X_pred = model.predict(test_data_reshaped)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred_df = pd.DataFrame(X_pred, columns=test_data.columns)
X_pred_df.index = test_data.index
scored = pd.DataFrame(index=test_data.index)
Xtest = test_data_reshaped.reshape(test_data_reshaped.shape[0], test_data_reshaped.shape[2])
#anormallik skorları belirlenir
scored['Loss_mae'] = np.mean(np.abs(X_pred_df-Xtest), axis=1)
scored['Threshold'] = 0.012
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
scored.head()


X_pred_train = model.predict(train_data_reshaped)
X_pred_train =X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[2])
X_pred_train_df = pd.DataFrame(X_pred_train, columns=train_data.columns)
X_pred_train_df.index = train_data.index
scored_train = pd.DataFrame(index=train_data.index)
Xtrain = train_data_reshaped.reshape(train_data_reshaped.shape[0], train_data_reshaped.shape[2])
scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-Xtrain), axis=1)
scored_train['Threshold'] = 0.012
scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
scored = pd.concat([scored_train, scored])

scored.plot(logy=True, figsize=(16, 9), ylim=[1e-2, 1e2], color=['blue', 'red'])
plt.title('Anomaly Detection', fontsize=16)
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Loss (MAE)', fontsize=14)
plt.show()

#model bilgilerini kaydediyoruz h5 formatında. Daha sonra eğitim eğitilmiş veri üzerinden devam edebilsin diye
model.save('my_model.keras')
print("Model saved")
'''












# Sonuçları yazdırın
#stratified_ornekler.to_json('DataSet.json', orient='records')

#print(stratified_ornekler.info())


