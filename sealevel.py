"""
Name: Elizabeth Vanegas
Email: Elizabeth.Vanegas11@myhunter.cuny.edu

Abstract: The project is about sea level rise in areas in NYC. Using linear regression to find the slope of sea level rise in NYC and predict future sea levels.
Also using linear models we are able to predict approximately the years it takes for sea level to reach an elevation point.

URL: https://elizabethvanegas11.wixsite.com/sealeveltrend
Resources:
https://inferentialthinking.com/chapters/15/2/Regression_Line.html
https://geopandas.org/en/stable/gallery/plotting_with_folium.html
http://www.maps-gps-info.com/how-elevation-is-determined/

DATA - 
Battery,NY -
https://tidesandcurrents.noaa.gov/sltrends/sltrends_station.shtml?stnid=8518750
https://tidesandcurrents.noaa.gov/sltrends/data/8518750_meantrend.csv
Atlantic Ocean -
https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/LSA_SLR_timeseries.php
https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/slr/slr_sla_atl_keep_txj1j2_90.csv
Elevation -
https://data.cityofnewyork.us/Transportation/Elevation-points/szwg-xci6
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
import seaborn as sns
import geopandas
import folium

# Display Data
#Graph for relay sea level rise
# 8518750_meantrend.csv - The Battery,NY dataset 
data = pd.read_csv('8518750_meantrend.csv', index_col= False)
print (data)
year = data['Year']
sea_levels = data[' Linear_Trend']
plt.plot(year, sea_levels)
plt.xlabel('Year')
plt.ylabel('Sea Level(Meters)')
plt.title('The Battery, NY')
plt.show()

#slr_sla_atl_keep_txj1j2_90.csv Atlantic Ocean dataset for TOPEX/Poseidon
data = pd.read_csv('slr_sla_atl_keep_txj1j2_90.csv', index_col= False,skiprows=5)
data.dropna()
print (data)
plt.scatter(data['year'],data['TOPEX/Poseidon'])
# plt.plot(year, sea_levels)
plt.xlabel('Year')
plt.ylabel('Sea Level(Meters)')
plt.title('Atlantic mean sea level from TOPEX/Poseidon')
plt.show()

#Graph for Atlantic Ocean sea level rise

data = pd.read_csv('slr_sla_atl_keep_txj1j2_90.csv', index_col= False,skiprows=5)
year = data['year']
data.plot(x='year',  grid=True)
plt.xlabel('Year')
plt.ylabel('Changes in mean sea level(mm)')
plt.title('Atlantic mean sea level from TOPEX/Poseidon')
plt.show()


#Linear Regression

#Computes linear trend line
def compute_r_line(xes,yes):
	xes = pd.Series(xes)
	yes = pd.Series(yes)
	sd_x = np.std(xes)
	sd_y = np.std(yes)
	r = xes.corr(yes)
	m = r*sd_y/sd_x
	b = yes[0] - m * xes[0]
	return m , b

# Linear Regression for The Battery,NY Linear Trend 
battery = pd.read_csv('8518750_meantrend.csv', index_col= False)
m, b = compute_r_line(battery['Year'],battery[' Linear_Trend'])
print(m,b)
xes = np.array([1850,2020])
yes = m*xes + b
plt.scatter(battery['Year'],battery[' Linear_Trend'])
plt.plot(xes,yes,color='red')
plt.title(f'Regression line of Linear Trend m = {m:{4}.{2}}')
plt.xlabel('Year')
plt.ylabel('Sea Level(Meters)')
plt.show()

# results: slope is an increase of 0.0028788708253429784 meters/per year
#2.8788708253429784 mm per year

# print((0.0028788708253429784)/12) 
#0.00023990590211191487 meters/per month
#0.23990590211191487 mm/per month

# Linear regression of Topex/Poiseidon - missing data 
atlantic = pd.read_csv('slr_sla_atl_keep_txj1j2_90.csv', index_col= False,skiprows=5)
m, b = compute_r_line(atlantic['year'],atlantic['TOPEX/Poseidon'])
print(m,b)
xes = np.array([1990,2007])
yes = m*xes + b
plt.scatter(atlantic['year'],atlantic['TOPEX/Poseidon'])
plt.plot(xes,yes,color='red')
plt.title(f'Regression line of Linear Trend m = {m:{4}.{2}} and y-intercept = {b:{4}.{2}}')
plt.xlabel('Year')
plt.ylabel('Sea Level(Meters)')
plt.show()


# Predictions

#Predictor for year 
def predictor_year(meters):
	file = pd.read_csv('8518750_meantrend.csv', index_col= False)
	file = file.dropna()
	regr = linear_model.LinearRegression()
	regr.fit(file[[' Linear_Trend']], file['Year'])
	# print (f'Predicted value: {regr.predict([[2.3951184]])[0]}')
	return regr.predict([[meters]])[0]


# #Predictor for meter level rise 
def predictor_searise(year):
	file = pd.read_csv('8518750_meantrend.csv', index_col= False)
	file = file.dropna()
	regr = linear_model.LinearRegression()
	regr.fit(file[['Year']], file[' Linear_Trend'])
	# print (f'Predicted value: {regr.predict([[7380]])[0]}')
	return regr.predict([[year]])[0]
# Predicted value for year 2025: 0.09710759090098442
# Predicted value for year 2030: 0.11150194502769928
# Predicted value for year 2050: 0.16907936153455871

# extract coordinates 
def extractLatLon(df):
	strip_ = df.strip("POINT ( )")
	latitude,longitude = strip_.split()
	return (eval(latitude),eval(longitude))

#Prediction with elevation samples from NYC data
elevation  = pd.read_csv('ELEVATION.csv', index_col= False)
sample_row = elevation.sample()
lat_lon = sample_row.loc[sample_row.index[0], 'the_geom']
coords = extractLatLon(lat_lon) #extract coordinates
elevate_sample = sample_row.loc[sample_row.index[0], 'ELEVATION']
elevate_meter = elevate_sample/3.28 # convert feet to meters

# since sea level rises is approximately 0.0028788708253429784 meters per year we will divide that with the elevation 
# to see how many years for the sea level to rise to the sample corrdinate to compare with predictor year
# m if from Linear Regression for The Battery,NY Linear Trend 
battery = pd.read_csv('8518750_meantrend.csv', index_col= False)
m, b = compute_r_line(battery['Year'],battery[' Linear_Trend'])
years_predict = predictor_year(elevate_meter/m) # calculates how many years it will take sea level to reach the elevation level
# print(elevate_sample)
print('According to Battery NY data:')
print(f'Sea Level will reach elevation of {elevate_meter} meters at coordinates:{coords} in {years_predict} years')
print(f'approximately in {round(predictor_year(elevate_meter))}') # calculates the year it will take sea level to reach the elevation level
# ex: Sea Level will reach elevation of 60.50290805030488 meters at coordinates:(-73.93297517430554, 40.85958026855029) in 7301837.144598751 years
# approximately in 23007


#creates folium map of sample coordinates
lon,lat = extractLatLon(lat_lon) 
map = folium.Map(location = [lat,lon], tiles = "Stamen Terrain", zoom_start = 14)
map.add_child(folium.Marker(location = [lat,lon]))
map.save('map.html')


