#6-27-21_LoRaRescue_Function1

#Revision for Function

#Trilateration and K-Means Program 
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from kneed import KneeLocator
from scipy.optimize import fsolve

import serial
import time

import math
from datetime import datetime as dt
import csv

#############################################
arduino = serial.Serial('com5', 115200)
ts = time.localtime() #update time

#Define variables for use

# distanceA = list()
# distanceB = list()
# distanceC = list()

# okA = 0
# okB = 0
# okC = 0
# ok = 0

# phoneA = 0
# phoneB = 1
# phoneC = 2

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

def listenForData():

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

    while ok == 0: #will wait until ok is 1. 'ok' will only be 1 when A B C are matching.

        arduino_raw_data = arduino.readline() #read serial data
        decoded_data = str(arduino_raw_data.decode("utf-8")) #convert to utf-8
        data = decoded_data.replace('\n','') #remove \n in the decoded data

        gatewayID = data[:1] #get gateway ID
        dataID = data[1:2] #get data ID

        data = data[2:] #get data

        if gatewayID == 'A':

            filename="gatewayA.csv" #set to write to gatewayA.csv

            if dataID == '1':
                phoneA = data
                print("\nPhone at Gateway A: 0" + phoneA)
            elif dataID == '2':
                distanceA.append(float(data))
            elif dataID == '3':

                # 1) Get time
                ts = time.localtime() #update time
                timeA = time.strftime("%X", ts) #set timeA to current time
                print("timeA: " + timeA)
                dtn = str(dt.now())
                dtn = dtn[0:19]
                dtn = dtn.replace(':',';')

                # 2) Put  distanceA values to distanceAf and clear distanceA for reuse
                distanceAf = distanceA
                del distanceA
                distanceA = list()
                distanceAf = np.delete(distanceAf,len(distanceAf)-1)
                distanceAf = np.delete(distanceAf,len(distanceAf)-1)
                # Convert to distance
                print("RSSI = ")
                print(distanceAf)
                rssiRawA = distanceAf
                distanceAf = distConv(distanceAf)
                print("DistanceAf = ")
                print(distanceAf)
                #print("Length of DistanceAf is: ")
                #print(len(distanceAf))
                okA = 1

                file = open(fileName, "a") #append timedata to the file
                file.write(timeA+"\n") #write timeA to csv file. The 3 is the dataID.
        
        elif gatewayID == 'B':
        
            fileName="gatewayB.csv" #set to write to gatewayB.csv

            if dataID == '1':
                phoneB = data
                print("\nPhone at Gateway B: 0" + phoneB)
            elif dataID == '2':
                distanceB.append(float(data))
            elif dataID == '3':
                if phoneB == phoneA:
                    
                    # 1) Set timeB == timeA
                    timeB = timeA
                    print("timeB: " + timeB)
                    
                    # 2) Put  distanceB values to distanceBf and clear distanceB for reuse
                    distanceBf = distanceB
                    del distanceB
                    distanceB = list()
                    distanceBf = np.delete(distanceBf,len(distanceBf)-1)
                    distanceBf = np.delete(distanceBf,len(distanceBf)-1)
                    # Convert to distance
                    print("RSSI = ")
                    print(distanceBf)
                    rssiRawB = distanceBf
                    distanceBf = distConv(distanceBf)
                    print("DistanceBf = ")
                    print(distanceBf)
                    #print("Length of DistanceBf is: ")
                    #print(len(distanceBf))
                    okB = 1

                    file = open(fileName, "a") #append timedata to the file
                    file.write(timeB+"\n") #write timeB to csv file. The 3 is the dataID.

                else:
                    del distanceB
                    distanceB = list()
                    print("phoneB is not the same as phoneA. Data will be discarded.")

        elif gatewayID == 'C':

            fileName="gatewayC.csv" #set to write to gatewayC.csv

            if dataID == '1':
                phoneC = data
                print("\nPhone at Gateway C: 0" + phoneC)
            if dataID == '2':
                distanceC.append(float(data))
            if dataID == '3':
                if phoneC == phoneB == phoneA:
                    # 1) Set timeC == timeB
                    timeC = timeB
                    print("timeC: " + timeC)
                    # 2) Put  distanceB values to distanceBf and clear distanceB for reuse
                    distanceCf = distanceC
                    del distanceC
                    distanceC = list()
                    distanceCf = np.delete(distanceCf,len(distanceCf)-1)
                    distanceCf = np.delete(distanceCf,len(distanceCf)-1)
                    # Convert to distance
                    print("RSSI = ")
                    print(distanceCf)
                    rssiRawC = distanceCf
                    distanceCf = distConv(distanceCf)
                    print("DistanceCf = ")
                    print(distanceCf)
                    #print("Length of DistanceCf is: ")
                    #print(len(distanceCf))
                    okC = 1

                    file = open(fileName, "a") #append timedata to the file
                    file.write(timeC+"\n") #write timeB to csv file. The 3 is the dataID.

                else:
                    del distanceC
                    distanceC = list()
                    print("phoneC is not the same as phoneB and phoneA. Data will be discarded.")
        
        #Writing to CSV -- Dito nangyayari yung writing to .csv ng Cellphone number and Distance values. Sa taas yung timestamp.
        file = open(fileName, "a") #append the data to the file
        file.write(data) #write data to csv file

        if okA == 1 & okB == 1 & okC == 1:
            ok = 1
            print("\nA, B, and C distances with the same phone number successfully obtained!\n")

    return distanceAf, distanceBf, distanceCf #return the variables

distanceAf, distanceBf, distanceCf = listenForData()



    