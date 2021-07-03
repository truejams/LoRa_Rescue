#6-27-21_LoRaRescue_Function1

#Revision for Function

# Import Code
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from kneed import KneeLocator
from scipy.optimize import *
import serial
import time
import math
from datetime import datetime as dt
import csv

# Variable declarations
port = 'com19'
baud = 115200

ts = time.localtime() #update time

#Define variables for use
distanceA = list()
distanceB = list()
distanceC = list()
okA = 0
okB = 0
okC = 0
ok = 0
phoneA = 0
phoneB = 1
phoneC = 2

###### CHANGE THIS FOR YOUR DIRECTORY
################################################################
save_destination = "C:\\Users\\Benj\\Desktop\\LoRa_Rescue\\6-27-21_LoRaRescue_Function1.py\\"

# Distance calculation constants
n = 3.2
dro = 1.5
roRSSI = -32

def distConv(rssi):
    dist = list()
    for i in range(int(len(rssi))):
        dist.append(pow(10,((roRSSI-int(rssi[i]))/(10*n)))*dro)
    return dist
def trilatEqn(z):
    x = z[0]
    y = z[1]
    w = z[2]

    F = np.empty((3))
    F[0] = ((x-xg[0])**2) + ((y-yg[0])**2) - (dA**2)
    F[1] = ((x-xg[1])**2) + ((y-yg[1])**2) - (dB**2)
    F[2] = ((x-xg[2])**2) + ((y-yg[2])**2) - (dC**2)
    return F

#Start of Program

print('\nProgram started\n')
print('Listening to the specified COM Port')
################################################################

# To use the function, call it with the ff line :
# distanceAf, distanceBf, distanceCf = collectSerialdata()
# Function Declarations
def listenForData(port,baud):
    #Define variables for use
    print("listening to port "+str(port)+" at "+str(baud))
    arduino = serial.Serial(port, baud)
    rssiA = list()
    rssiB = list()
    rssiC = list()
    phoneA = 0
    phoneB = 0
    phoneC = 0
    okA = 0
    okB = 0
    okC = 0
    ok = 0

    while ok == 0: #will wait until ok is 1. 'ok' will only be 1 when A B C are matching.
        arduino_raw_data = arduino.readline() #read serial data
        decoded_data = str(arduino_raw_data.decode("utf-8")) #convert to utf-8
        data = decoded_data.replace('\n','') #remove \n in the decoded data
        data = data.replace('\r','')
        gatewayID = data[:1] #get gateway ID
        dataID = data[1:2] #get data ID
        data = data[2:] #get data
        if gatewayID == 'A':
            if dataID == '1':
                phoneA = data
                print("\nReceiving Gateway A: 0" + phoneA)
            elif dataID == '2':
                rssiA.append(float(data))
            elif dataID == '3':
                ts = time.localtime() #update time
                timeA = time.strftime("%X", ts) #set timeA to current time
                print("timeA: " + timeA)
                dtn = str(dt.now())
                dtn = dtn[0:19]
                dtn = dtn.replace(':',';')
                rssiA = np.delete(rssiA,len(rssiA)-1)
                rssiA = np.delete(rssiA,len(rssiA)-1)
                okA = 1
        elif gatewayID == 'B':
            if dataID == '1':
                phoneB = data
                print("\nReceiving Gateway B: 0" + phoneB)
            elif dataID == '2':
                rssiB.append(float(data))
            elif dataID == '3':
                ts = time.localtime() #update time
                timeB = time.strftime("%X", ts) #set timeA to current time
                print("timeB: " + timeB)
                rssiB = np.delete(rssiB,len(rssiB)-1)
                rssiB = np.delete(rssiB,len(rssiB)-1)
                okB = 1
        elif gatewayID == 'C':
            if dataID == '1':
                phoneC = data
                print("\nReceiving Gateway C: 0" + phoneC)
            if dataID == '2':
                rssiC.append(float(data))
            if dataID == '3':
                ts = time.localtime() #update time
                timeC = time.strftime("%X", ts) #set timeA to current time
                print("timeC: " + timeC)
                rssiC = np.delete(rssiC,len(rssiC)-1)
                rssiC = np.delete(rssiC,len(rssiC)-1)
                okC = 1
        # Write to CSV, note if data matches
        if phoneA == phoneB == phoneC and okA == 1 and okB == 1 and okC == 1:
            with open(save_destination+'rawData.csv', mode='a') as logs:
                logswrite = csv.writer(logs, dialect='excel', lineterminator='\n')
                logswrite.writerow(['Phone','Time','Gateway A','Gateway B','Gateway C'])
                for i in range(len(rssiA)):
                    logswrite.writerow([phoneA,timeA,rssiA[i],rssiB[i],rssiC[i]])
            start_dt = dt.strptime(timeA, '%H:%M:%S')
            end_dt = dt.strptime(timeC, '%H:%M:%S')
            diff = abs(end_dt - start_dt)
            print("\nA, B, and C received successfully with interval of "+str(diff))
            ok = 1
        elif okA == 1 and okB == 1 and okC == 1:
            print("\nError: Data mismatch, dumping date into error.csv")
            with open(save_destination+'error.csv', mode='a') as logs:
                logswrite = csv.writer(logs, dialect='excel', lineterminator='\n')
                logswrite.writerow(['Time','Gateway A','Gateway B','Gateway C'])
                for i in range(len(rssiA)):
                    logswrite.writerow([phoneA,timeA,rssiA[i],rssiB[i],rssiC[i]])
            rssiA = list()
            rssiB = list()
            rssiC = list()
    return rssiA, rssiB, rssiC, dtn, phoneA #return the variables



    
