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

from pyproj import CRS, Transformer
import pandas as pd
import seaborn as sns
import folium
from selenium import webdriver
import serial
import time
from datetime import datetime as dt
from datetime import timedelta as td
import os
import pyrebase
from sklearn.cluster import DBSCAN
import json

# Variable declarations
port = 'COM3'
baud = 115200

ts = time.localtime() #update time

#Define variables for use
phone = 0
dist = list()
diff = list()

###### CHANGE THIS FOR YOUR DIRECTORY
save_destination = "C:\\LoRa_Rescue\\10-30-21_Data\\"

# Distance calculation constants
# Change based on desired coefficient
n = list()
nmin = 0.1
nmax = 20
ninterval = 0.1
dro = 1.5
roRSSI = -32

tdist = 0.0

#Trilateration calculation constants
# Gateway Node Coordinate (Cartesian; don't touch) ############ DO NOT TOUCH
xg = 0
yg = 0

# Gateway Node Position (GPS coordinates decimal)
longg = 14.6650923
latg = 120.9720347

# Actual Node Coordinates (Cartesian; dist. in meters) ############ DO NOT TOUCH
xAct = 0            #Target x-coordinate
yAct = 2.5          #Target y-coordinate

# Actual Mobile node Node Position (GPS Coordinates Decimal)
longAct = 14.6657547
latAct = 120.9712846

# Firebase Web App Configuration
LoraRescueStorage = {'apiKey': "AIzaSyAN2jdAfGBhbPz446Lho_Jmu2eysU6Hvqw",
    'authDomain': "lora-rescue.firebaseapp.com",
    'databaseURL': "https://lora-rescue-default-rtdb.asia-southeast1.firebasedatabase.app",
    'projectId': "lora-rescue",
    'storageBucket': "lora-rescue.appspot.com",
    'messagingSenderId': "295122276311",
    'appId': "1:295122276311:web:68ce5d4d4cd6763103c592",
    'measurementId': "G-MCPTP8HPLK"}

def serialListener(port,baud):
    #Define variables for use
    print("Listening to port "+str(port)+" at "+str(baud))
    arduino = serial.Serial(port, baud)
    rssiA = list()
    rssiB = list()
    rssiC = list()
    phoneA = 0
    phoneB = 0
    phoneC = 0
    done= 0
    ok = [0,0,0,0]
    
    while done == 0:
        arduino_raw_data = arduino.readline()
        decoded_data = str(arduino_raw_data.decode("utf-8")) #convert to utf-8
        rawData = decoded_data.replace('\n','') #remove \n in the decoded data
        rawData = rawData.replace('\r','')
        gatewayID = rawData[:1] #get gateway ID
        if gatewayID != 'A': continue
        tempData = rawData[1:] #get data
        dataSplit = list()
        dataSplit = tempData.split()
        phone = dataSplit[0]
        rssi = dataSplit[1].replace("\x00","")
        rssiTemp = rssi
        if not(rssiTemp.replace("-","").isnumeric()): continue
        dtn = str(dt.now())
        dtn = dtn[0:19]
        dateNow = dtn[0:10]
        timeNow = dtn[11:19]
        print(dtn + " Received Gateway " + gatewayID + ": +63" + phone + " with RSSI: " + rssi)

        if gatewayID == 'A':
            ok[0] = 1
            phoneA = phone
            temprssiA = rssi
        elif gatewayID == 'B':
            ok[1] = 1
            phoneB = phone
            temprssiB = rssi
        elif gatewayID == 'C':
            ok[2] = 1
            phoneC = phone
            temprssiC = rssi
        else:
            print("Unrecognized Serial Data: " + rawData)

        # Save to Database
        firebase = pyrebase.initialize_app(LoraRescueStorage)
        db = firebase.database()
        if sum(ok) == 1:
            for i in range(3): ok[i] = 0
            ok[3], timePrev = checkDatabase(dateNow,timeNow,phone)
            timeNow = timePrev
            rssiAlist = db.child(dateNow).child(timeNow).child("Calibration RSSI Values").child("RSSI Gateway A").get().val()
            if rssiAlist != None:
                rssiAlist.append(temprssiA)
            else:
                rssiAlist = [temprssiA]
            db.child(dateNow).child(timeNow).child("Calibration RSSI Values").child("RSSI Gateway A").set(rssiAlist)
            
            if ok[3] == 1:
                ok = [0,0,0,0]
                done = 1

    return rssiAlist, phone

def checkDatabase(dateNow,timeNow,phone):
    check = 0
    timePrev = timeNow
    n = 30 #This is range for time
    firebase = pyrebase.initialize_app(LoraRescueStorage)
    db = firebase.database()
    databaseEntries = db.child(dateNow).get() #retreive the realtime database datapoints
    if databaseEntries.val() == None:
        check = 0
        timePrev = timeNow + " 0" + phone
        return check, timePrev
    x = json.dumps(list(databaseEntries.val().items())) #convert ordered list to json
    df = pd.read_json(x) #convert json to pandas dataframe for processing
    entries = df.iloc[:, 0].tolist() #convert pandas dataframe to list of strings
    for i in range(len(entries)):
        checkDatePhone = entries[i].split()
        if checkDatePhone[1] == "0"+phone:
            tPrev = dt.strptime(checkDatePhone[0],'%H:%M:%S')
            tNow = dt.strptime(timeNow,'%H:%M:%S') - td(minutes=n)
            if tPrev > tNow:
                timePrev = str(tPrev)[11:19]
            else:
                timePrev = timeNow
    timePrev = timePrev + " 0" + phone
    databaseEntries = db.child(dateNow).child(timePrev).child("Calibration RSSI Values").get()
    if databaseEntries.val() == None:
        check = 0
        return check, timePrev
    df = pd.read_json(json.dumps(list(databaseEntries.val().items())))
    entries = df.iloc[0, 1]
    print(dateNow + " " + timePrev + " has " + str(len(entries)) + " entries")
    if len(entries) >= 50:
        check = 1
    return check, timePrev

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
# rssi, phone = listenForData(port,baud)
rssi, phone = serialListener(port, baud)
for i in range(len(rssi)): rssi[i] = int(rssi[i])
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