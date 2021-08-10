import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit

from termcolor import colored
# from timer import start_timer
from timer import stop_timer

df = pd.read_csv("C:/Users/metek/Desktop/auto.csv")
df = df.replace("?", np.NAN)
df.isnull().sum()

# 1. ŞIK
# start_timer()

start_time1 = timeit.default_timer()
start_time_full = timeit.default_timer()

symboling = df['symboling']
normalizedLoss = df['normalized-losses'] = df['normalized-losses'].astype(float)
make = df['make']
fuelType = df['fuel-type']
aspiration = df['aspiration']
numOfDoors = df['num-of-doors']
bodyStyle = df['body-style']
driveWheels = df['drive-wheels']
engineLocation = df['engine-location']
wheelBase = df['wheel-base']
length = df['length']
width = df['width']
height = df['height']
# curbWeigh = df['curb-weigh']
engineType = df['engine-type']
numOfCylinders = df['num-of-cylinders']
engineSize = df['engine-size']
fuelSystem = df['fuel-system']
bore = df['bore'] = df['bore'].astype(float)
stroke = df['stroke'] = df['stroke'].astype(float)
compressionRatio = df['compression-ratio']
horsepower = df['horsepower'] = df['horsepower'].astype(float)
peakRpm = df['peak-rpm'] = df['peak-rpm'].astype(float)
cityMpg = df['city-mpg']
highwayMpg = df['highway-mpg']
price = df['price'] = df['price'].astype(float)

# Mean Değerleri
print("Ortalama Symboling:", np.mean(symboling))
print("Ortalama Normalized Losses:", np.mean(normalizedLoss))
print("Ortalama Wheel Base:", np.mean(wheelBase))
print("Ortalama Length:", np.mean(length))
print("Ortalama Width:", np.mean(width))
print("Ortalama Height:", np.mean(height))
# print("Ortalama Curb Weigh:", np.mean(curbWeigh))
print("Ortalama Bore:", np.mean(bore))
print("Ortalama Stroke:", np.mean(stroke))
print("Ortalama horsepower:", np.mean(horsepower))
print("Ortalama Peak RPM:", np.mean(peakRpm))
print("Ortalama City MPG:", np.mean(cityMpg))
print("Ortalama Highway MPG:", np.mean(highwayMpg))
print("Ortalama Price:", np.mean(price))

# Median Değerleri
print("\nSymboling Medyan:", np.median(symboling))
print("Normalized Losses Medyan:", np.median(normalizedLoss))
print("Wheel Base Medyan:", np.median(wheelBase))
print("Length Medyan:", np.median(length))
print("Width Medyan:", np.median(width))
print("Height Medyan:", np.median(height))
# print("Curb Weigh Medyan:", np.median(curbWeigh))
print("Bore Medyan:", np.median(bore))
print("Stroke Medyan:", np.median(stroke))
print("Horsepower Medyan:", np.median(horsepower))
print("Peak RPM Medyan:", np.median(peakRpm))
print("City MPG Medyan:", np.median(cityMpg))
print("Highway MPG Medyan:", np.median(highwayMpg))
print("Price Medyan:", np.median(price))

# Max Değerleri
print("\nSymboling Max:", np.max(symboling))
print("Normalized Losses Max:", np.max(normalizedLoss))
print("Wheel Base Max:", np.max(wheelBase))
print("Length Max:", np.max(length))
print("Width Max:", np.max(width))
print("Height Max:", np.max(height))
# print("Curb Weigh Max:", np.max(curbWeigh))
print("Bore Max:", np.max(bore))
print("Stroke Max:", np.max(stroke))
print("Horsepower Max:", np.max(horsepower))
print("Peak RPM Max:", np.max(peakRpm))
print("City MPG Max:", np.max(cityMpg))
print("Highway MPG Max:", np.max(highwayMpg))
print("Price Max:", np.max(price))

# Min Değerleri
print("\nSymboling Min:", np.min(symboling))
print("Normalized Losses Min:", np.min(normalizedLoss))
print("Wheel Base Min:", np.min(wheelBase))
print("Length Min:", np.min(length))
print("Width Min:", np.min(width))
print("Height Min:", np.min(height))
# print("Curb Weigh Min:", np.min(curbWeigh))
print("Bore Min:", np.min(bore))
print("Stroke Min:", np.min(stroke))
print("Horsepower Min:", np.min(horsepower))
print("Peak RPM Min:", np.min(peakRpm))
print("City MPG Min:", np.min(cityMpg))
print("Highway MPG Min:", np.min(highwayMpg))
print("Price Min:", np.min(price))

# STD değerleri
print("\nSymboling Standart Sapma:", np.std(symboling))
print("Normalized Losses Standart Sapma:", np.std(normalizedLoss))
print("Wheel Base Standart Sapma:", np.std(wheelBase))
print("Length Standart Sapma:", np.std(length))
print("Width Standart Sapma:", np.std(width))
print("Height Standart Sapma:", np.std(height))
# print("Curb Weigh Standart Sapma:", np.std(curbWeigh))
print("Bore Standart Sapma:", np.std(bore))
print("Stroke Standart Sapma:", np.std(stroke))
print("Horsepower Standart Sapma:", np.std(horsepower))
print("Peak RPM Standart Sapma:", np.std(peakRpm))
print("City MPG Standart Sapma:", np.std(cityMpg))
print("Highway MPG Standart Sapma:", np.std(highwayMpg))
print("Price Standart Sapma:", np.std(price))

elapsed1 = timeit.default_timer() - start_time1
print(colored('\n****************************************************************', 'green'), )
print(colored(f'1. Kod bloğunda geçen süre {elapsed1}', 'green'), )
print(colored('****************************************************************', 'green'), )
# 2. ŞIK
start_time2 = timeit.default_timer()
print("\n----------------------------------------------")
print("\nVeri Setindeki Araba Markası Sayısı = ", df['make'].unique().size)
print("\nMarkaların İsimleri = ", df['make'].unique())

# 3. ŞIK

print("\n----------------------------------------------")
print("\nEN PAHALI 5 ARABA MARKASI")
carMake = df.groupby('make')
priceDf = carMake['price'].max()
print(priceDf.sort_values(ascending=False).head(5))
print("\nEN UCUZ 5 ARABA MARKASI")
print(priceDf.sort_values(ascending=True).head(5))

# 4. ŞIK

print("\n----------------------------------------------")
diesel = df[df['fuel-type'] == 'diesel']
dieselSedan = diesel.groupby(['body-style']).get_group('sedan')
print("\nDizel ve sedan olan araç sayısı = ", len(dieselSedan))

# 5. ŞIK

print("\n----------------------------------------------")
print("\nBEYGİR GÜCÜ EN DÜŞÜK 5 ARABA")
hp_car_max = df.groupby('horsepower')
hp_df_max = hp_car_max['horsepower', 'make', 'drive-wheels', 'wheel-base'].max()
print(hp_df_max.head(5))
print("\nBEYGİR GÜCÜ EN YÜKSEK 5 ARABA")
print(hp_df_max.tail(5))

# 6. ŞIK

print("\n----------------------------------------------")
gas = df[df['fuel-type'] == 'gas']
gasFront = gas.groupby(['engine-location']).get_group('front')
print("\nMotoru önde ve benzin kullanan araç sayısı = ", len(gasFront))

# 7. ŞIK

print("\n----------------------------------------------")
hatch = df.loc[df.groupby((df['body-style'] == 'hatchback')).groups[1]]
sedan = df.loc[df.groupby((df['body-style'] == 'sedan')).groups[1]]
wagon = df.loc[df.groupby((df['body-style'] == 'wagon')).groups[1]]
plt.figure(figsize=(45, 25))
plt.title("7. SORU", fontsize=30)
plt.xlabel('Araba Üreticileri', fontsize=30)
plt.ylabel('Araba Sayıları', fontsize=30)
plt.hist(df.make, bins=60, label="Markanın Toplam Araç Sayısı", color="red")
plt.hist(hatch.make, bins=60, label="Hatchback Sayısı", color="purple")
plt.hist(sedan.make, bins=60, label="Sedan Sayısı", color="cyan")
plt.hist(wagon.make, bins=60, label="Wagon Sayısı", color="green")
plt.legend(loc="upper center", ncol=2, fontsize=18)
plt.savefig("auto.pdf")
plt.savefig("auto.eps")
plt.savefig("auto.svg")
plt.show()

# B SORUSU
# 1. ŞIK

df_2 = pd.read_csv("C:/Users/metek/Desktop/auto.csv")
ml = df_2['fuel-type']
mlCopy = df_2.copy()
mlCopy.drop(columns=["fuel-type", "make", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location",
                     "engine-type", "num-of-cylinders", "fuel-system", "symboling"], axis=1, inplace=True)

from sklearn.neighbors import KNeighborsClassifier

knn1 = KNeighborsClassifier(n_neighbors=10)
knn2 = KNeighborsClassifier(n_neighbors=5)
knn3 = KNeighborsClassifier(n_neighbors=10)
knn4 = KNeighborsClassifier(n_neighbors=5)
mlCopy.replace('?', 0, inplace=True)

from sklearn.model_selection import train_test_split

x1_egitim, x1_test, y1_egitim, y1_test = train_test_split(mlCopy, ml, test_size=0.9, train_size=0.1, random_state=42)
x2_egitim, x2_test, y2_egitim, y2_test = train_test_split(mlCopy, ml, test_size=0.9, train_size=0.1, random_state=42)
x3_egitim, x3_test, y3_egitim, y3_test = train_test_split(mlCopy, ml, test_size=0.6, train_size=0.4, random_state=42)
x4_egitim, x4_test, y4_egitim, y4_test = train_test_split(mlCopy, ml, test_size=0.7, train_size=0.3, random_state=42)

model1 = knn1.fit(x1_egitim, y1_egitim)
model2 = knn2.fit(x2_egitim, y2_egitim)
model3 = knn1.fit(x3_egitim, y3_egitim)
model4 = knn1.fit(x4_egitim, y4_egitim)

model1.predict([[160, 100, 170, 70, 55, 3000, 130, 3, 3, 5, 110, 5600, 18, 20, 17000]])
model2.predict([[160, 100, 170, 70, 55, 3000, 130, 3, 3, 5, 110, 5600, 18, 20, 17000]])
model3.predict([[160, 100, 170, 70, 55, 3000, 130, 3, 3, 5, 110, 5600, 18, 20, 17000]])
model4.predict([[160, 100, 170, 70, 55, 3000, 130, 3, 3, 5, 110, 5600, 18, 20, 17000]])

score1 = model1.score(x1_egitim, y1_egitim)
score2 = model2.score(x2_egitim, y2_egitim)
score3 = model3.score(x3_egitim, y3_egitim)
score4 = model4.score(x4_egitim, y4_egitim)

print("\n----------------------------------------------")
print("1. TEST SKORU: ", score1)
print("2. TEST SKORU: ", score2)
print("3. TEST SKORU: ", score3)
print("4. TEST SKORU: ", score4)

elapsed2 = timeit.default_timer() - start_time2
print(colored('\n****************************************************************', 'green'), )
print(colored(f'Geri kalan Kod bloğunda geçen süre {elapsed2}', 'green'), )
print(colored('****************************************************************', 'green'), )

elapsed_full = timeit.default_timer() - start_time_full
print(colored('\n****************************************************************', 'blue'), )
print(colored(f'Komple kod için geçen süre {elapsed_full * 1000} ms', 'blue'), )
print(colored('****************************************************************', 'blue'), )

stop_timer()
