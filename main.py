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


log_file_path1 = 'C:/Users/zeyze/Desktop/tasarımLog/FGT60E4Q16043370db.log20231219.json'
log_file_path2 = 'C:/Users/zeyze/Desktop/tasarımLog/FGT60E4Q16043370db.log20231220.json'
log_file_path3 = 'C:/Users/zeyze/Desktop/tasarımLog/FGT60E4Q16043370db.log20231221.json'
#C:/Users/zeyze/Desktop/tasarım-proje/log-girdi/FGT60E4Q16043370db.log20230804.json

log_file_paths = [log_file_path1, log_file_path2, log_file_path3]

# Boş bir DataFrame oluşturun
combined_df = pd.DataFrame()


for file_path in log_file_paths:

    #dosyadan verileri okuyor
    with open(file_path, 'r') as file:
        json_data = file.read()

    #okunan verileri data frame oluşturup içine koyuyor.
    #eski sürümlerde read_json fonksiyonu direkt dosyayı okuyabiliyordu artık stringIO tipini parametre alıyor.
    df = pd.read_json(StringIO(json_data))


    #kalmasını istedigim sutunlar(time,policytype,srccountry ı cikardim unutma)
    columns = ['action','app','appcat','datetime','devtype','dstcountry','dstintf','dstip','dstport','duration',
               'policyid','rcvdbyte','sentbyte','service','srcintf','srcip','srcmac',
               'subtype']


    #baska bir data frame olusturup icine bu sutunları verileriyle beraber atıyoruz. içinde bu sutunların bulunmadıgı verilere sutunları ekler degerleri null yapar.
    df_newColums= df[columns].copy()

    filter_column = 'subtype'
    filter_value= ['forward','system','app-ctrl','webfilter']
    #sdwan,wireless,ssl eklenebilir


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
        # Belirlediğiniz örnek sayısını ayarlayın (örneğin 1)
        ornek_grup = log.sample(n=1, random_state=42)
        ornekler.append(ornek_grup)

    # Stratified örnekleri birleştirin
    stratified_ornekler = pd.concat(ornekler)


    #datetime sutununu tekrar jsona uygun hale getiriyor
    #stratified_ornekler["datetime"] = stratified_ornekler["datetime"].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))


    print("info stratified_ornekler")
    print(stratified_ornekler.info())

    combined_df = pd.concat([combined_df, stratified_ornekler], ignore_index=True)

    print("info combined_df")
    print(combined_df.info())


#sayısallaştırma yapabilmek için sütunları düzenleyelim

#dict tipindeki  verileri null yapıyor.
for column_name in combined_df.columns:
    if column_name != "datetime":
        combined_df[column_name] = combined_df[column_name].apply(lambda x: None if isinstance(x, dict) else x)

# app sütununu düzenler
#stratified_ornekler['app'] = stratified_ornekler['app'].apply(lambda x: str(x).split('/')[0] if pd.notna(x) else x)

#FrequencyEncoded ile sayısallaştırma işlemi
for objectColumn in combined_df.select_dtypes(include='object').columns:
    if objectColumn and objectColumn != "datetime":
        #frekansları hesaplar
        category_freq = combined_df[objectColumn].value_counts(normalize=True)
        #frekans degerlerine gore sütunu düzenler
        combined_df[objectColumn] = combined_df[objectColumn].map(category_freq)


print("info combined_df sayisal")
print(combined_df.info())

column_types = combined_df.dtypes
print(column_types)



#null deger çok ise ve kategorik sütunsa ort, kategorik sütnlarda genellikle mod, en çok tekrar edenle diğerleri arasında uçurum varsa medyan, int değerlerde kesinlikle ort değil aykırı değerler etkilemesin diye
medyan_columns = ['dstport']
mod_columns = ['action','dstcountry','dstintf','dstip','policyid','service','srcintf','srcip',
               'subtype']
ort_columns = ['app','devtype','srcmac','appcat','duration','rcvdbyte','sentbyte']
int_columns = ['dstport','duration','policyid','rcvdbyte','sentbyte']


#ortalama,mod,medyan ile eksik veri tamamlama, seçili sütunlara normalizasyon uygulama
for column in combined_df.columns:
    if column in ort_columns:
        mean_value = combined_df[column].mean(skipna=True)
        combined_df[column].fillna(mean_value, inplace=True)
    if column in mod_columns:
        modes = mode(combined_df[column], nan_policy='omit')  # mode fonksiyonu ile mod değeri bulunuyor
        mode_value = modes.mode.max() #en çok tekrar eden birden fazla değer varsa en büyğünü al
        combined_df[column].fillna(mode_value, inplace=True)
    if column in medyan_columns:
        medyan_value = np.nanmedian(combined_df[column])  # nanmedian fonksiyonu ile medyan değeri bulunuyor
        combined_df[column].fillna(medyan_value, inplace=True)
    if column in int_columns:
        scaler = MinMaxScaler()
        combined_df[column] = scaler.fit_transform(combined_df[column].values.reshape(-1, 1))

    print(column)
    print(combined_df[column].max())
    print(combined_df[column].min())


#pca nın gerekliliğine ve boyutuna karar vermek için datetime içermeyen veri seti oluşturduk
columns = ['action','app','appcat','devtype','dstcountry','dstintf','dstip','dstport','duration',
           'policyid','rcvdbyte','sentbyte','service','srcintf','srcip','srcmac',
           'subtype']
pcaAnalyzeMatris = combined_df[columns].copy()



#pca nın gerekliliğine karar vermek için şartları kontrol ediyoruz

#VIF degerini kontrol ediyoruz VIF>5 ise pca uygulanması mantıklı
def calculate_vif(dataframe):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = dataframe.columns
    vif_data["VIF"] = [variance_inflation_factor(dataframe.values, i) for i in range(dataframe.shape[1])]
    return vif_data

#calculate_vif(pcaAnalyzeMatris)

#korelasyon matrisi oluşturuyoruz.
def calculate_correlation_matrix(dataframe):
    correlation_matrix = dataframe.corr()
    return correlation_matrix

#calculate_correlation_matrix(pcaAnalyzeMatris)

#koveryans matrisi oluşturuyoruz. Düşük varyans zayıf ilişki pca uygulanabilir.
def calculate_covariance_matrix(dataframe):
    cov_matrix = np.cov(dataframe, rowvar=False)
    cov_df = pd.DataFrame(cov_matrix, columns=dataframe.columns, index=dataframe.columns)
    return cov_df

#calculate_covariance_matrix(pcaAnalyzeMatris)

#yamaç eğrisi grafiği ile pca nın boyutuna karar veriyoruz.
def plot_pca_variance(dataframe):

    # PCA modelini oluştur
    pca = PCA()
    pca.fit(dataframe)

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

#plot_pca_variance(pcaAnalyzeMatris)


# Tarih/saat sütununu string'ten datetime'a dönüştürme
#stratified_ornekler['datetime'] = pd.to_datetime(stratified_ornekler['datetime'],format='%Y.%m.%d.%H.%M.%S')

# datetime sütununu sakla
hidden_column = combined_df['datetime']
combined_df.drop('datetime', axis=1, inplace=True)


#pca yı yapıyoruz

# PCA modelini oluştur (svd_solver='full' yani korelasyon matrisini kullanarak)
pca = PCA(n_components=10)
pca_result = pca.fit_transform(combined_df)

# PCA sonuçlarını yeni bir DataFrame'e ekle
result_df = pd.DataFrame(data=pca_result, columns=[f'PC{i}' for i in range(1, 11)])

# datetime sütununu tekrar ekleyerek sonuçları birleştir
pca_df = pd.concat([hidden_column.reset_index(drop=True), result_df], axis=1)

merged_data = pd.DataFrame()

# Set 'datetime_column' as index
pca_df = pca_df.set_index('datetime')

merged_data = pd.concat([merged_data, pca_df], axis=1)


print("merge data info")
print(merged_data.info())


def print_explained_variance(pca_model):
    # Açıklanan varyans oranlarını al
    explained_variance_ratio = pca_model.explained_variance_ratio_

    # Her bir bileşenin açıklanan varyans oranını ekrana yazdır
    for i, ratio in enumerate(explained_variance_ratio, 1):
        print(f'Bileşen {i}: {ratio:.4f}')

    # Toplam varyansı bulun
    total_variance = np.sum(explained_variance_ratio)
    print(f"Toplam varyans: {total_variance * 100:.2f}%")
    print(f'Toplam Açıklanan Varyans: {total_variance:.4f}')

#print_explained_variance(pca)

def visualize_pca_results(pca_model, pca_result):
    # Açıklanan varyans oranlarını çiz
    plt.subplot(1, 3, 3)
    plt.bar([f'PC{i}' for i in range(1, len(pca_model.explained_variance_ratio_) + 1)],
            pca_model.explained_variance_ratio_)
    plt.title('Explained Variance ')
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

#visualize_pca_results(pca, pca_result)


#normal dagilima sahip olup olmadigini gormek icin grafikler
def plot_qq(data, column_name):
    sm.qqplot(data[column_name], line='45', fit=True)
    plt.title(f'{column_name} QQ Plot')
    plt.show()

#plot_qq(merged_data, 'sentbyte')


def plot_histogram(data, column_name, bins=7):
    plt.figure(figsize=(10, 6))
    sns.histplot(x=column_name, data=data, bins=bins, kde=False, color='blue')
    plt.title(f'{column_name} Histogram')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

# Örnek kullanım
#plot_histogram(merged_data, 'sentbyte')


# Veriyi eğitim ve test olarak bölme
split_ratio = 0.7  # Eğitim verisinin oranı, bu oranı ihtiyacınıza göre ayarlayabilirsiniz
split_index = int(split_ratio * len(merged_data))  # Verinin kaçıncı indeksine kadar eğitim verisi alınacak hesaplanıyor

train_data = merged_data[:split_index]   # Eğitim verisi, verinin başından belirlenen indekse kadar olan kısmı
test_data = merged_data[split_index:]   # Test verisi, belirlenen indeksten verinin sonuna kadar olan kısmı




# Eğitim verisi
print("Eğitim Verisi:")
print(train_data.head())

# Test verisi
print("Test Verisi:")
print(test_data.head())


#verilerin zaman içindeki değişimini göteren grafik
def plot_pca_time_series(data, components=['PC1', 'PC2', 'PC3', 'PC4','PC5','PC6','PC7','PC8','PC9','PC10']):
    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)

    for component in components:
        ax.plot(data[component], label=component)

    plt.legend(loc='lower left')
    ax.set_title('Time Series of PCA Components', fontsize=16)
    plt.show()

#plot_pca_time_series(train_data)
#plot_pca_time_series(test_data)

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

    # İkinci LSTM katmanı: 4 birim, aktivasyon fonksiyonu 'tanh', zaman dizisini döndürme
    L2 = LSTM(4, activation='tanh', return_sequences=False)(L1)

    #zaman adımındaki örüntüler için. Burada tekrarlama sayısını mı 10 dk periyot olarak ayarlıycaz yoksa zaman adımını mı
    # Giriş dizisini tekrar etme katmanı (RepeatVector)
    L3 = RepeatVector(X.shape[1])(L2)

    # Üçüncü LSTM katmanı: 4 birim, aktivasyon fonksiyonu 'tanh', zaman dizisini döndür
    L4 = LSTM(4, activation='tanh', return_sequences=True)(L3)

    # Dördüncü LSTM katmanı: 16 birim, aktivasyon fonksiyonu 'tanh', zaman dizisini döndür
    L5 = LSTM(16, activation='tanh', return_sequences=True)(L4)

    # Çıkış katmanını tanımla: Zamanla dağıtılmış (TimeDistributed) bir yoğun (Dense) katman
    output = TimeDistributed(Dense(X.shape[2], activation='tanh'))(L5)

    # Modeli oluştur
    model = Model(inputs=inputs, outputs=output)
    return model


#autoencoder model oluştur
model = autoencoder_model(train_data_reshaped)
#model derleniyor
#optimizer için alternatifler:adam,sgd,rmsprop #loss için alternatifler: mae,mse,huber
model.compile(optimizer='adam', loss='mae')
#modelin özeti görüntülenir
model.summary()

nb_epochs = 300
#her bir eğitim iterasyonunda kullanılacak örnek sayısını belirleyen bir parametre
batch_size = 32
#validation_split eğitim sırasında ayrılmış bir doğrulama seti kullanılmasını sağlar
#eğitim veri seti eğitilir. eğitim sonunda eğitim kayıplarına ve doğrulama kayıplarına bu değişken üzerinden erişilebilir 
history = model.fit(train_data_reshaped, train_data_reshaped, epochs=nb_epochs, batch_size=batch_size,validation_split=0.1).history



#eğitim sürecinde elde edilen kayıpların grafiğini çizmek için kullanılır.Doğrulama örnekleri ve eğitim örnekleri için
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae) ')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()


# eğitim seti üzerinde modelin tahminlerini kullanarak(eğitim sonrasında) kayıpların dağılımını görselleştirir
X_pred = model.predict(train_data_reshaped)
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
plt.show()


#modelin test veri kümesi üzerindeki performansını değerlendirir. 
#İlk olarak, model kullanılarak test veri kümesinin tahminleri (X_pred) elde edilir 
#ardından bu tahminlerle gerçek test verisi (Xtest) arasındaki (kayıp) değerleri hesaplanır. 
X_pred = model.predict(test_data_reshaped)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred_df = pd.DataFrame(X_pred, columns=test_data.columns)
X_pred_df.index = test_data.index
scored = pd.DataFrame(index=test_data.index)
Xtest = test_data_reshaped.reshape(test_data_reshaped.shape[0], test_data_reshaped.shape[2])
#anormallik skorları belirlenir
scored['Loss_mae'] = np.mean(np.abs(X_pred_df-Xtest), axis=1)
#anomali için eşik değeri
scored['Threshold'] = 0.19
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
#scored her bir zaman damgası için kayıp değerini, eşik değerini ve bu gözlemin anormal olup olmadığını içerir.
scored.head()


scored.plot(logy=True, figsize=(16, 9), ylim=[1e-2, 1e2], color=['blue', 'red'])
plt.title('Anomaly Detection', fontsize=16)
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Loss (MAE)', fontsize=14)
plt.show()




X_pred_train = model.predict(train_data_reshaped)
X_pred_train =X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[2])
X_pred_train_df = pd.DataFrame(X_pred_train, columns=train_data.columns)
X_pred_train_df.index = train_data.index
scored_train = pd.DataFrame(index=train_data.index)
Xtrain = train_data_reshaped.reshape(train_data_reshaped.shape[0], train_data_reshaped.shape[2])
scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-Xtrain), axis=1)
scored_train['Threshold'] = 0.19
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













# Sonuçları yazdırın
#stratified_ornekler.to_json('DataSet.json', orient='records')

#print(stratified_ornekler.info())


