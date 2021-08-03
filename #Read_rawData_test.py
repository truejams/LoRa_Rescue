#Read_rawData_test
import pandas as pd
import numpy as np

rawdataread = pd.read_csv(r"C:\Users\Benj\Desktop\LoRa_Rescue\8-3-21_PandasTests\rawData.csv", header=0)
#r"" specifies that it is a string. This is the location of the csv to be read
#header=0 means headers at row 0

startrow = 60 #starting header row
endrow = 118 #ending row of the dataset

rawdatalim = rawdataread[startrow-1:endrow-1] #limit for which columns to read.

rawRSSIAread = rawdatalim['Gateway A'].to_numpy(float) #reads column with Gateway A header and then converts into numpy float array
rawRSSIBread = rawdatalim['Gateway B'].to_numpy(float) #reads column with Gateway B header and then converts into numpy float array
rawRSSICread = rawdatalim['Gateway C'].to_numpy(float) #reads column with Gateway C header and then converts into numpy float array

print(rawdatalim) #print rows that were read
print(rawRSSIAread) #print Gateway A RSSI values
print(rawRSSIBread) #print Gateway B RSSI values
print(rawRSSICread) #print Gateway C RSSI values