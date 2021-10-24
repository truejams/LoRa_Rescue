# LoRa Rescue Calibration Code

# Build v0.1
# This code has been tested indoors with LOS environment
# The mobile node was placed 2.5 meters away
# The result showed an optimal path loss exponent of 9.9

# Build v0.2
# Incorporated haversine formula for calibration

# To use the code no editing of the arduino code will be necessary
# Simply plug the gateway and transmit data from the mobile node
# It is noted that only gateway A shall be used for calibration.

from numpy.lib.nanfunctions import nanmax, nanmin
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from kneed import KneeLocator
from scipy.optimize import *
import serial
import time
import math
from math import radians, cos, sin, asin, sqrt
from datetime import datetime as dt
import csv

# Variable declarations
port = 'COM6'
baud = 115200

ts = time.localtime() #update time

#Define variables for use
phone = 0
dist = list()
diff = list()

###### CHANGE THIS FOR YOUR DIRECTORY
save_destination = "C:\\Users\\Benj\\Desktop\\LoRa_Rescue\\10-23-21_Data\\"

# Distance calculation constants
# Change based on desired coefficient
n = list()
nmin = 0.1
nmax = 50
ninterval = 0.1
dro = 1.5
roRSSI = -32

tdist = 0.0

#Trilateration calculation constants
# Gateway Node Coordinate (Cartesian; don't touch) ############ DO NOT TOUCH
xg = 0
yg = 0

# Gateway Node Position (GPS coordinates decimal)
longg = 14.6650704
latg = 120.9720111

# Actual Node Coordinates (Cartesian; dist. in meters) ############ DO NOT TOUCH
xAct = 0            #Target x-coordinate
yAct = 2.5          #Target y-coordinate

# Actual Mobile node Node Position (GPS Coordinates Decimal)
longAct = 14.6654448
latAct = 120.9715698

# Function Declarations
def listenForData(port,baud):
    #Define variables for use
    print("\nListening to "+str(port)+" at "+str(baud))
    arduino = serial.Serial(port, baud)
    rssi = list()
    phone = 0
    okA = 0
    ok = 0

    while ok == 0: #will wait until ok is 1. 'ok' will only be 1 when A B C are matching.
        arduino_raw_data = arduino.readline() #read serial data
        decoded_data = str(arduino_raw_data.decode("utf-8")) #convert to utf-8
        # print(decoded_data)
        data = decoded_data.replace('\n','') #remove \n in the decoded data
        data = data.replace('\r','')
        gatewayID = data[:1] #get gateway ID
        dataID = data[1:2] #get data ID
        data = data[2:] #get data
        if gatewayID == 'A':
            if dataID == '1':
                phone = data
                print("Receiving gateway with phone: 0" + phone)
            elif dataID == '2':
                rssi.append(float(data))
                print(data)
            elif dataID == '3':
                # ts = time.localtime() #update time
                # time = time.strftime("%X", ts) #set timeA to current time
                # print("time: " + time)
                # dtn = str(dt.now())
                # dtn = dtn[0:19]
                # dtn = dtn.replace(':',';')
                rssi = np.delete(rssi,len(rssi)-1)
                rssi = np.delete(rssi,len(rssi)-1)
                okA = 1
        # Write to CSV, note if data matches
        if phone != 0 and okA == 1:
            print("\nRSSI data received")
            ok = 1
    return rssi, phone #return the variables

def rssiToDist(rssi,n,dro,roRSSI):
    distA = list()
    dist = pow(10,((roRSSI-int(rssi))/(10*n)))*dro
    return dist
 
# Start calibration
actDist = sqrt(((xg-xAct)**2)+((yg-yAct)**2))
def haversine(lat1, lon1, lat2, lon2):

    miles = 3959.87433
    meters = 6372.8*1000

    R = meters

    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))

    distance = R * c

    return distance

actDist = haversine(latg,longg,latAct,longAct)

print("\nMobile Node is "+ str(actDist) +" meters away gateway A")
rssi, phone = listenForData(port,baud)
averssi = sum(rssi)/len(rssi)
print("\nSweeping n with range "+ str(nmin) +" to "+ str(nmax) +" with interval "+ str(ninterval))
for i in range(int(nmin/ninterval),int((nmax+ninterval)/ninterval)):
    temp = i*ninterval
    n.append(temp)
    tdist = rssiToDist(averssi,temp,dro,roRSSI)
    # print(tdist)
    dist.append(tdist)
    diff.append(abs(tdist-actDist))
optN = n[diff.index(min(diff))]
print("The optimal coefficient is "+ str(optN) +" with distance of "+ str(dist[diff.index(min(diff))]))

# Save findings to calibration.csv
with open(save_destination+'calibration.csv', mode='a') as logs:
    logswrite = csv.writer(logs, dialect='excel', lineterminator='\n')
    logswrite.writerow(['Phone','aveRSSI','n','Distance'])
    for i in range(len(n)):
        logswrite.writerow([phone,averssi,n[i],dist[i]])
print("Saved data to calibration.csv\n")