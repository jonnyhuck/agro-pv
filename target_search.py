from pandas import read_csv
from geopandas import read_file

# load files
# countries = read_file("C:\\Users\\mzdssjh8\\Dropbox\\Manchester\\Teaching\\30552-Understanding_GIS\\Practicals\\repos\\code\\data\\natural-earth\\ne_50m_admin_0_countries.shp")
countries = read_file("C:\\Users\\mzdssjh8\\Dropbox\\Manchester\\Research\\AgroPV\\data\\corrected_countries.shp")
targets = read_csv("C:\\Users\\mzdssjh8\\Dropbox\\Manchester\\Research\\AgroPV\\Target 0691.csv")

# get ISO codes
country_codes = set(countries['ISO_A3'].to_list())
target_codes = set(targets['ISO_3'].to_list())

# find codes in the targets that are missing from country dataset 
print(target_codes.difference(country_codes))
print("\n---\n")

# get the countries not included in the Target file 
print(len(countries.index))
print(len(country_codes))
print(len(target_codes))
for iso in country_codes.difference(target_codes):
    row = countries[countries.ISO_A3 == iso].iloc[0]
    print(f"{row.NAME} ({row.ISO_A3})")