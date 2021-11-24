# Import Libraries
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from scipy.optimize import *
from math import radians, cos, sin, asin, sqrt
import csv
from pyproj import CRS, Transformer
import pandas as pd
import seaborn as sns
import folium
import serial
import time
from datetime import datetime as dt
from datetime import timedelta as td
import os
import pyrebase
from sklearn.cluster import DBSCAN
import json
from sklearn.neighbors import NearestNeighbors

# Variable Declaration
################## CHANGE THIS ACCORDINGLY ##################  
# Benjamin's Directory
# save_destination = "C:\\LoRa_Rescue\\11-21-21_Tests\\"
# Ianny's Directory
# save_destination = "D:\\Users\\Yani\\Desktop\\LoRa Rescue Data\\"
# Greg's Directory
save_destination = "C:\\LoRa_Rescue\\"

# Change Current Working Directory in Python
os.chdir(save_destination)

# Arduino Configuration
################## CHANGE THIS ACCORDINGLY ##################  
port = "COM3"
baud = 115200

# Firebase Web App Configuration
LoraRescueStorage = {'apiKey': "AIzaSyAN2jdAfGBhbPz446Lho_Jmu2eysU6Hvqw",
    'authDomain': "lora-rescue.firebaseapp.com",
    'databaseURL': "https://lora-rescue-default-rtdb.asia-southeast1.firebasedatabase.app",
    'projectId': "lora-rescue",
    'storageBucket': "lora-rescue.appspot.com",
    'messagingSenderId': "295122276311",
    'appId': "1:295122276311:web:68ce5d4d4cd6763103c592",
    'measurementId': "G-MCPTP8HPLK"}

# Read RawData.csv Configuration
# In excel, the first row is treated as Row 0
    # Basically, subtract 1 from excel row number
################## CHANGE THIS ACCORDINGLY ##################  
startrow = 1
endrow = 58

# RSSI to Distance calculation constants
################## CHANGE THIS ACCORDINGLY ##################  
n = 2.8
nA = nB = nC = n
# nA = 2.3
# nB = 2.7
# nC = 2.5
dro = 1
roRSSI = -30

# Trilateration calculation constants
# GNode GPS Coordinates
# Format: A B C
################## CHANGE THIS ACCORDINGLY ##################  
latg = np.array([14.6651047,14.6671611,14.6664435])
longg = np.array([120.9720628,120.9695632,120.9704663])

# GNode Cartesian Coordinates
# Format: A B C
xg = np.array([0,0,0])
yg = np.array([0,0,0])

# Actual Mobile Node GPS Coordinates
################## CHANGE THIS ACCORDINGLY ##################  
# A - 11-13-21
latAct = np.array([14.6667305])
longAct = np.array([120.9700698])

# B - 11-13-21
# latAct = np.array([14.6656659])
# longAct = np.array([120.9713092])

# Actual Mobile Node Cartesian Coordinates
xAct = np.array([0]) #Target x-coordinate
yAct = np.array([0]) #Target y-coordinate

################## CHANGE THIS ACCORDINGLY ##################
# Circle Trilateration Points
points = 100

# Tolerance filter error margin
################## CHANGE THIS ACCORDINGLY ##################  
errorTolerance = 1000

# DBSCAN calculation constants
################## CHANGE THIS ACCORDINGLY ##################
kNeighbors = 3 # kNeighbors = 2*Dimensions -1 = 2*2 -1 = 3    
minPts = 4 # MinPts = k + 1 = 3 + 1 = 4

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
                timeA = str(dt.now())[0:19]
                timeA = timeA.replace(':','-')
                print("timeA: " + timeA)
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
                timeB = str(dt.now())[0:19]
                timeB = timeB.replace(':','-')
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
                timeC = str(dt.now())[0:19]
                timeC = timeC.replace(':','-')
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
            start_dt = dt.strptime(timeA[11:19], '%H-%M-%S')
            end_dt = dt.strptime(timeC[11:19], '%H-%M-%S')
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

    return rssiA, rssiB, rssiC, timeA, phoneA #return the variables

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
            temprssiB = str(int(rssi) + 9)
        elif gatewayID == 'C':
            ok[2] = 1
            phoneC = phone
            temprssiC = rssi
        else:
            print("Unrecognized Serial Data: " + rawData)

        # Save to Database
        firebase = pyrebase.initialize_app(LoraRescueStorage)
        db = firebase.database()
        if sum(ok) == 3 and phoneA == phoneB and phoneB == phoneC:
            for i in range(3): ok[i] = 0
            ok[3], timePrev = checkDatabase(dateNow,timeNow,phone)
            timeNow = timePrev
            rssiAlist = db.child(dateNow).child(timeNow).child("Raw RSSI Values").child("RSSI Gateway A").get().val()
            if rssiAlist != None:
                rssiAlist.append(temprssiA)
            else:
                rssiAlist = [temprssiA]
            db.child(dateNow).child(timeNow).child("Raw RSSI Values").child("RSSI Gateway A").set(rssiAlist)
            rssiBlist = db.child(dateNow).child(timeNow).child("Raw RSSI Values").child("RSSI Gateway B").get().val()
            if rssiBlist != None:
                rssiBlist.append(temprssiB)
            else:
                rssiBlist = [temprssiB]
            db.child(dateNow).child(timeNow).child("Raw RSSI Values").child("RSSI Gateway B").set(rssiBlist)
            rssiClist = db.child(dateNow).child(timeNow).child("Raw RSSI Values").child("RSSI Gateway C").get().val()
            if rssiClist != None:
                rssiClist.append(temprssiC)
            else:
                rssiClist = [temprssiC]
            db.child(dateNow).child(timeNow).child("Raw RSSI Values").child("RSSI Gateway C").set(rssiClist)

            if ok[3] == 1:
                ok = [0,0,0,0]
                done = 1
            
    # Write to CSV
    databaseEntries = db.child(dateNow).child(timePrev).child("Raw RSSI Values").get()
    df = pd.read_json(json.dumps(list(databaseEntries.val().items())))
    rssiA = df.iloc[0, 1]
    rssiB = df.iloc[1, 1]
    rssiC = df.iloc[2, 1]
    temptime = timeNow.split()
    dtn = dateNow + " " + temptime[0]
    dtn = dtn.replace(':','-')
    with open(save_destination+'rawData.csv', mode='a') as logs:
        logswrite = csv.writer(logs, dialect='excel', lineterminator='\n')
        logswrite.writerow(['Phone','Time','Gateway A','Gateway B','Gateway C'])
        for i in range(len(rssiA)):
            logswrite.writerow([phone,dtn,rssiA[i],rssiB[i],rssiC[i]])
    return rssiA, rssiB, rssiC, dtn, phone

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
    databaseEntries = db.child(dateNow).child(timePrev).child("Raw RSSI Values").get()
    if databaseEntries.val() == None:
        check = 0
        return check, timePrev
    df = pd.read_json(json.dumps(list(databaseEntries.val().items())))
    entries = df.iloc[0, 1]
    print(dateNow + " " + timePrev + " has " + str(len(entries)) + " entries")
    if len(entries) >= 50:
        check = 1
    return check, timePrev

def importCSV(save_destination, startrow, endrow):
    rawdataread = pd.read_csv(save_destination + 'rawData.csv', header=0)
    #r"" specifies that it is a string. This is the location of the csv to be read
    #header=0 means headers at row 0

    rawdatalim = rawdataread[startrow-1:endrow-1] #limit for which columns to read.

    phone = rawdatalim['Phone'].to_list()[0] #reads 1st column with Phone header
    dtn  = rawdatalim['Time'].to_list()[0] #reads 1st column with Time header
    rssiA = rawdatalim['Gateway A'].to_numpy(float) #reads column with Gateway A header and then converts into numpy float array
    rssiB = rawdatalim['Gateway B'].to_numpy(float) #reads column with Gateway B header and then converts into numpy float array
    rssiC = rawdatalim['Gateway C'].to_numpy(float) #reads column with Gateway C header and then converts into numpy float array
    
    return rssiA, rssiB, rssiC, dtn, phone

def importDatabase(date, phoneTime):
    temp = phoneTime.split()
    time = temp[0]
    phone = temp[1]
    firebase = pyrebase.initialize_app(LoraRescueStorage)
    db = firebase.database()
    databaseEntries = db.child(date).child(phoneTime).child("Raw RSSI Values").get()
    df = pd.read_json(json.dumps(list(databaseEntries.val().items())))
    rssiA = df.iloc[0, 1]
    rssiB = df.iloc[1, 1]
    rssiC = df.iloc[2, 1]
    databaseEntries = db.child(date).child(phoneTime).child("Actual Data").get()
    df = pd.read_json(json.dumps(list(databaseEntries.val().items())))
    mobileLatLong = df.iloc[1, 1].split()
    latAct = np.array([float(mobileLatLong[0])])
    longAct = np.array([float(mobileLatLong[1])])
    latAct,longAct = cartToGPS(latAct,longAct)

    databaseEntries = db.child(date).child(phoneTime).child("Basic Raw Information").get()
    df = pd.read_json(json.dumps(list(databaseEntries.val().items())))
    gnodeA = df.iloc[3, 1].split()
    gnodeB = df.iloc[4, 1].split()
    gnodeC = df.iloc[5, 1].split()
    dtn = date + " " + time
    dtn = dtn.replace(':','-')
    latg = np.array([float(gnodeA[0]),float(gnodeB[0]),float(gnodeC[0])])
    longg = np.array([float(gnodeA[1]),float(gnodeB[1]),float(gnodeC[1])])
    latg,longg = cartToGPS(latg,longg)
    phone = phone[1:]
    print('GNode Latitudes: ' + str(latg))
    print('GNode Longitudes: '+ str(longg))
    return rssiA, rssiB, rssiC, dtn, phone, latg, longg, latAct, longAct

def rssiToDist(rssi,n,dro,roRSSI):
    dist = list()
    for i in range(len(rssi)):
        dist.append(pow(10,((roRSSI-float(rssi[i]))/(10*n)))*dro)

    return dist

def drawCircle(xg,yg,rA,rB,rC,points):
    intersect = [[0,[0,0]],[0,[0,0]],[0,[0,0]]]
    r = [rA,rB,rC]
    x = [[],[],[]]
    y = [[],[],[]]
    pi = 3.14
    for i in range(3):
        for j in range(points):
            x[i].append(r[i]*cos(2*pi*j/points)+xg[i])
            y[i].append(r[i]*sin(2*pi*j/points)+yg[i])
    if (rA - rB)**2 <= (xg[0] - xg[1])**2 + (yg[0] - yg[1])**2 and (xg[0] - xg[1])**2 + (yg[0] - yg[1])**2 <= (rA + rB)**2:
        xint, yint = get_intersections(xg[0], yg[0], rA, xg[1], yg[1], rB)
        intersect[0] = [1,[xint,yint]]
    if (rB - rC)**2 <= (xg[1] - xg[2])**2 + (yg[1] - yg[2])**2 and (xg[1] - xg[2])**2 + (yg[1] - yg[2])**2 <= (rB + rC)**2:
        xint, yint = get_intersections(xg[1], yg[1], rB, xg[2], yg[2], rC)
        intersect[1] = [1,[xint,yint]]
    if (rA - rC)**2 <= (xg[0] - xg[2])**2 + (yg[0] - yg[2])**2 and (xg[0] - xg[2])**2 + (yg[0] - yg[2])**2 <= (rA + rC)**2:
        xint, yint = get_intersections(xg[0], yg[0], rA, xg[2], yg[2], rC)
        intersect[2] = [1,[xint,yint]]
    return x,y,intersect

def get_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d=sqrt((x1-x0)**2 + (y1-y0)**2)
    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 
        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
        x = [x3,x4]
        y = [y3,y4]
        return x,y

def trilaterateCircle(xCirc,yCirc,intersect,points):
    deltaDist = [10000,10000,10000]
    dist = [0,0,0]
    x = [0,0,0]
    y = [0,0,0]
    for i in range(3):
        for j in range(points):
            for k in range(points):
                if i <= 1:
                    dist[i] = sqrt((xCirc[i][j]-xCirc[i+1][k])**2)+((yCirc[i][j]-yCirc[i+1][k])**2)
                    if dist[i] < deltaDist[i]:
                        deltaDist[i] = dist[i]
                        x[i] = (xCirc[i][j]+xCirc[i+1][k])/2
                        y[i] = (yCirc[i][j]+yCirc[i+1][k])/2
                elif i == 2:
                    dist[i] = sqrt((xCirc[i][j]-xCirc[0][k])**2)+((yCirc[i][j]-yCirc[0][k])**2)
                    if dist[i] < deltaDist[i]:
                        deltaDist[i] = dist[i]
                        x[i] = (xCirc[i][j]+xCirc[0][k])/2
                        y[i] = (yCirc[i][j]+yCirc[0][k])/2
    for i in range(3):
        if intersect[i][0] == 1 and i < 2:
            dist1 = sqrt(((intersect[i][1][0][0]-x[2])**2)+((intersect[i][1][1][0]-y[2])**2))
            dist2 = sqrt(((intersect[i][1][0][1]-x[2])**2)+((intersect[i][1][1][1]-y[2])**2))
            if dist1 < dist2:
                x[i] = intersect[i][1][0][0]
                y[i] = intersect[i][1][1][0]
            else:
                x[i] = intersect[i][1][0][1]
                y[i] = intersect[i][1][1][1]
        elif intersect[i][0] == 1 and i == 2:
            dist1 = sqrt(((intersect[i][1][0][0]-x[0])**2)+((intersect[i][1][1][0]-y[0])**2))
            dist2 = sqrt(((intersect[i][1][0][1]-x[0])**2)+((intersect[i][1][1][1]-y[0])**2))
            if dist1 < dist2:
                x[i] = intersect[i][1][0][0]
                y[i] = intersect[i][1][1][0]
            else:
                x[i] = intersect[i][1][0][1]
                y[i] = intersect[i][1][1][1] 
    xTrilat = sum(x)/3
    yTrilat = sum(y)/3
    return xTrilat,yTrilat

def trilaterate(distanceAf,distanceBf,distanceCf,xg,yg):
    A = -2*xg[0]+2*xg[1]
    B = -2*yg[0]+2*yg[1]
    C = distanceAf**2-distanceBf**2-xg[0]**2+xg[1]**2-yg[0]**2+yg[1]**2
    D = -2*xg[1]+2*xg[2]
    E = -2*yg[1]+2*yg[2]
    F = distanceBf**2-distanceCf**2-xg[1]**2+xg[2]**2-yg[1]**2+yg[2]**2
    x = (C*E-F*B)/(E*A-B*D)
    y = (C*D-A*F)/(B*D-A*E)

    return x,y

def tolFilter(x,y,xAve,yAve,errorTolerance):
    i = 0
    while i != 50:
        if i == len(y):
            i = 50
            continue
        e = 0
        dist = np.sqrt(((xAve-x[i])**2)+((yAve-y[i])**2))
        if dist >= errorTolerance:
            #print(str(i)+" - Deleted"+' '+str(x[i])+' '+str(y[i]))
            x = np.delete(x,i)
            y = np.delete(y,i)
        else:
            #print(str(i)+' '+str(x[i])+' '+str(y[i]))
            i += 1

    return x,y

def kmeansOptimize(data):
    # Compute for inertias for every possible number of clusters
    inertia = [] #aka Sum of Squared Distance Errors
    for i in range(1,len(data)):
        kmeans = KMeans(n_clusters=i).fit(data)
        inertia.append(kmeans.inertia_)

    # Determine optimal Number of Clusters based on Elbow
    elbow = KneeLocator(range(1,len(data)), inertia, curve='convex', direction='decreasing')

    # Perform K-means with elbow no. of clusters
    kmeans = KMeans(n_clusters=elbow.knee, n_init=5).fit(data)

    return kmeans, inertia, elbow

def dbscanOptimize(data, minPts, k):
    # Determine distances of each point to their nearest neighbor
    nNeighbor = NearestNeighbors(n_neighbors=k).fit(data) # reference point is included in n_neighbors
    nNeighborDistance, nNeighborIndices = nNeighbor.kneighbors(data)
    nNeighborDistance = np.sort(nNeighborDistance, axis=0)[:,1] # Sort by columns/x values

    # Determine optimal epsilon based on Elbow
    dbElbow = KneeLocator(range(len(data)), nNeighborDistance, curve='convex', direction='increasing', online=True)

    if dbElbow.knee_y == 0:
        dbElbow.knee_y = 10**-3

    # Perform DBSCAN with epsilon elbow 
    dbscan = DBSCAN(eps=dbElbow.knee_y, min_samples=minPts).fit(data)

    return dbscan, nNeighborDistance, dbElbow

def distanceFormula(x1, y1, x2, y2):
    distance = np.sqrt(((x1-x2)**2)+((y1-y2)**2))

    return distance

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

    distance = np.array([R*c])

    return distance

def GPSToCart(lat,lon):
    # Convert GPS Coordinates to Cartesian Coordinates
    # For manual checking, refer to https://epsg.io/transform#s_srs=4326&t_srs=25391 
    # Defining CRS aka Coordinate Reference Systems
    inCRS = CRS.from_epsg(4326) #CRS of GPS that uses Latitude and Longitude Values
    outCRS = CRS.from_epsg(25391) #Luzon Datum of 1911 utilizing Transverse Mercator Project Map

    #Conversion from GPS Coordinates to Philippine Cartesian Coordinates
    GeoToCart = Transformer.from_crs(inCRS,outCRS)
    x,y = GeoToCart.transform(lat,lon) #Format should be (lat,lon) for epsg:4326
    
    return x,y

def cartToGPS(x,y):
    # Convert Cartesian Coordinates back to GPS Coordinates
    # For manual checking, refer to https://epsg.io/transform#s_srs=25391&t_srs=4326 
    # Defining CRS aka Coordinate Reference Systems
    inCRS = CRS.from_epsg(4326) #CRS of GPS that uses Latitude and Longitude Values
    outCRS = CRS.from_epsg(25391) #Luzon Datum of 1911 utilizing Transverse Mercator Project Map

    #Conversion from Philippine Cartesian back to GPS Coordinates Coordinates
    CartToGeo = Transformer.from_crs(outCRS,inCRS)
    lat, lon = CartToGeo.transform(x,y) #Format should be (x,y) for epsg:25391
    lat = list(lat)
    lon = list(lon)

    return lat, lon

def errorComp(x, y, xOld, yOld, xAct, yAct, kmeans, xAve, yAve, xAveOld, yAveOld, data):
    compVact = list()
    for i in range(len(x)):
        compVact.append(np.sqrt((x[i]-xAct)**2+(y[i]-yAct)**2))

    # K-means centroid vs. Average Point (dataset average)
    centVave = np.sqrt((kmeans.cluster_centers_[:,0]-xAve)**2+(kmeans.cluster_centers_[:,1]-yAve)**2)

    # Computed Position vs. K-means centroid
    compVcent = np.sqrt([(data[:,0]-kmeans.cluster_centers_[0,0])**2+(data[:,1]-kmeans.cluster_centers_[0,1])**2])
    for i in range(1,len(kmeans.cluster_centers_)):
        distance = np.sqrt([(data[:,0]-kmeans.cluster_centers_[i,0])**2+(data[:,1]-kmeans.cluster_centers_[i,1])**2])
        compVcent = np.append(compVcent,distance,axis=0)

    # Compute percentage increase/decrease from old to new trilateration 
    # Using distance average
    oldtriVact = distanceFormula(xOld,yOld,xAct,yAct)
    newtriVact = distanceFormula(np.array([x]),np.array([y]),xAct,yAct)
    # Using coordinate average
    # oldtriVact = distanceFormula(xAveOld,yAveOld,xAct,yAct)
    # newtriVact = distanceFormula(xAve,yAve,xAct,yAct)
    # Percentage increase/decrease formula
    oldtriVact = np.mean(oldtriVact)
    newtriVact = np.mean(newtriVact)
    triImprovement = ((newtriVact - oldtriVact) / (oldtriVact))*-100

    return compVact, centVave, compVcent, triImprovement

def firebaseUpload(firebaseConfig, localDir, cloudDir):
    # Initialize Firebase Storage
    firebase = pyrebase.initialize_app(firebaseConfig)
    storage = firebase.storage()

    # Upload files to Firebase Storage
    storage.child(cloudDir).put(localDir)


############ Kalman Filter Functions ##########

def kalman_block(x, P, s, A, H, Q, R):

    """
    Prediction and update in Kalman filter

    input:
        - signal: signal to be filtered
        - x: previous mean state
        - P: previous variance state
        - s: current observation
        - A, H, Q, R: kalman filter parameters

    output:
        - x: mean state prediction
        - P: variance state prediction

    """

    # check laaraiedh2209 for further understand these equations

    x_mean = A * x + np.random.normal(0, Q, 1)
    P_mean = A * P * A + Q

    K = P_mean * H * (1 / (H * P_mean * H + R))
    x = x_mean + K * (s - H * x_mean)
    P = (1 - K * H) * P_mean

    return x, P


def kalman_filter(signal, A, H, Q, R):

    """

    Implementation of Kalman filter.
    Takes a signal and filter parameters and returns the filtered signal.

    input:
        - signal: signal to be filtered
        - A, H, Q, R: kalman filter parameters

    output:
        - filtered signal

    """

    predicted_signal = []

    x = signal[0]                                 # takes first value as first filter prediction
    P = 0                                         # set first covariance state value to zero

    predicted_signal.append(x)
    for j, s in enumerate(signal[1:]):            # iterates on the entire signal, except the first element

        x, P = kalman_block(x, P, s, A, H, Q, R)  # calculates next state prediction

        predicted_signal.append(x)                # update predicted signal with this step calculation

    return predicted_signal

# Listen/Read for Data
# Retrieve RSSI data, date and time, and phone number

# Listen to COM port and check for errors
################## CHANGE THIS ACCORDINGLY ##################  
# rssiA, rssiB, rssiC, dtn, phoneA = listenForData(port,baud)
# rssiA, rssiB, rssiC, dtn, phoneA = serialListener(port,baud)

# Manually retrieve data from rawData.csv
################## CHANGE THIS ACCORDINGLY ##################  
# rssiA, rssiB, rssiC, dtn, phoneA = importCSV(save_destination, startrow, endrow)
# Format - Date: "2021-10-30" Time and Phone : "14:46:14 09976800632"
rssiA, rssiB, rssiC, dtn, phoneA, latg, longg, latAct, longAct =  importDatabase("2021-11-06", "17:06:34 09976500621")

# Compensation
for i in range(len(rssiB)):
    rssiA[i] = str(int(int(rssiA[i]) - 11))
    rssiB[i] = str(int(int(rssiB[i])))
    rssiC[i] = str(int(int(rssiC[i]) - 4))

################### RSSI Kalman ######################

rssiA_int = [int(i) for i in rssiA]
rssiB_int = [int(i) for i in rssiB]
rssiC_int = [int(i) for i in rssiC]

rssiA_kalman = kalman_filter(rssiA_int, A=1, H=1, Q=0.005, R=1)
rssiB_kalman = kalman_filter(rssiB_int, A=1, H=1, Q=0.005, R=1)
rssiC_kalman = kalman_filter(rssiC_int, A=1, H=1, Q=0.005, R=1)

# Convert RSSI to Distance
# distanceAf = rssiToDist(rssiA,nA,dro,roRSSI)
# distanceBf = rssiToDist(rssiB,nB,dro,roRSSI)
# distanceCf = rssiToDist(rssiC,nC,dro,roRSSI)

# Convert Kalman Filter RSSI to Distance
distanceAf = rssiToDist(rssiA_kalman,nA,dro,roRSSI)
distanceBf = rssiToDist(rssiB_kalman,nB,dro,roRSSI)
distanceCf = rssiToDist(rssiC_kalman,nC,dro,roRSSI)

# Convert GPS Coordinates to Cartesian Coordinates
xg,yg = GPSToCart(latg,longg)
xAct,yAct = GPSToCart(latAct,longAct)

# Trilateration Part of the Code
for i in range(len(distanceAf)):
    distanceAf[i] = float(distanceAf[i])
    distanceBf[i] = float(distanceBf[i])
    distanceCf[i] = float(distanceCf[i])
# Convert Distances from each GNode to numpy arrays
distanceAf = np.array(distanceAf)
distanceBf = np.array(distanceBf)
distanceCf = np.array(distanceCf)
# Get average distances
AfAve = sum(distanceAf)/len(distanceAf)
BfAve = sum(distanceBf)/len(distanceBf)
CfAve = sum(distanceCf)/len(distanceCf)
# Rotate Graph, comment if not needed
# xg, yg, xAct, yAct, notFlat = rotateGraph(xg, yg, xAct, yAct)

# Trilaterate Data
print("Trilaterating Data...")
x = list()
y = list()
for i in range(len(distanceAf)):
    distA = distanceAf[i]
    distB = distanceBf[i]
    distC = distanceCf[i]
    xCirc, yCirc, intersect = drawCircle(xg,yg,distA,distB,distC,points)
    xTrilat,yTrilat = trilaterateCircle(xCirc,yCirc,intersect,points)
    x.append(xTrilat)
    y.append(yTrilat)
xCircAve, yCircAve, inter = drawCircle(xg,yg,AfAve,BfAve,CfAve,points)
xAve,yAve = trilaterateCircle(xCircAve,yCircAve,inter,points)
print("Done Trilaterating!\n")

# Old Trilateration
xOld,yOld = trilaterate(distanceAf,distanceBf,distanceCf,xg,yg)
xAveOld,yAveOld = trilaterate(AfAve,BfAve,CfAve,xg,yg)

# print(x)
# print(y)

# Tolerance Filter  
x,y = tolFilter(x,y,xAve,yAve,errorTolerance)

# Mean Coordinates after Tolerance Filter
xFilt = x
yFilt = y
xFiltAve = np.mean(xFilt)
yFiltAve = np.mean(yFilt)

# Compute actual distances of GNodes to mobile node
################## CHANGE THIS ACCORDINGLY ##################  
# Use distance formula
# comp_distanceAf = distanceFormula(xAct, yAct, xg[0], yg[0])
# comp_distanceBf = distanceFormula(xAct, yAct, xg[1], yg[1])
# comp_distanceCf = distanceFormula(xAct, yAct, xg[2], yg[2])
# Use haversine formula
comp_distanceAf = haversine(latAct[0], longAct[0], latg[0], longg[0])
comp_distanceBf = haversine(latAct[0], longAct[0], latg[1], longg[1])
comp_distanceCf = haversine(latAct[0], longAct[0], latg[2], longg[2])

# Plot the data frequency of the gateways
fig = 1
plt.figure(fig)
distSeriesA = pd.Series(distanceAf).value_counts().reset_index().sort_values('index').reset_index(drop=True)
distSeriesA.columns = ['Distance [Meters]','Frequency']
distSeriesA['Distance [Meters]'] = distSeriesA['Distance [Meters]'].round()
distSeriesB = pd.Series(distanceBf).value_counts().reset_index().sort_values('index').reset_index(drop=True)
distSeriesB.columns = ['Distance [Meters]','Frequency']
distSeriesB['Distance [Meters]'] = distSeriesB['Distance [Meters]'].round()
distSeriesC = pd.Series(distanceCf).value_counts().reset_index().sort_values('index').reset_index(drop=True)
distSeriesC.columns = ['Distance [Meters]','Frequency']
distSeriesC['Distance [Meters]'] = distSeriesC['Distance [Meters]'].round()
figur, axes = plt.subplots(1,3, figsize=(18, 5))
axes[0].set_title(dtn + ' 0' + phoneA  + ' GNode A FD')
plots = sns.barplot(ax=axes[0],x="Distance [Meters]", y="Frequency", data=distSeriesA)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.1f'), 
                    (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                    size=9, xytext=(0, 8),
                    textcoords='offset points')

axes[1].set_title(dtn + ' 0' + phoneA  + ' GNode B FD')
plots = sns.barplot(ax=axes[1],x="Distance [Meters]", y="Frequency", data=distSeriesB)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.1f'), 
                    (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                    size=9, xytext=(0, 8),
                    textcoords='offset points')

axes[2].set_title(dtn + ' 0' + phoneA  + ' GNode C FD')
plots = sns.barplot(ax=axes[2],x="Distance [Meters]", y="Frequency", data=distSeriesC)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.1f'), 
                    (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                    size=9, xytext=(0, 8),
                    textcoords='offset points')

plt.setp(axes[0].get_xticklabels(), rotation=45, horizontalalignment='right')
plt.setp(axes[1].get_xticklabels(), rotation=45, horizontalalignment='right')
plt.setp(axes[2].get_xticklabels(), rotation=45, horizontalalignment='right')
plt.savefig(save_destination + dtn + ' 0' + phoneA + ' FrequencyDistribution.jpg', bbox_inches='tight')
fig += 1

#For the .py file exclusively, the DB graph unexpectedly mixes with the FD graph without plt.close()
plt.close()

# Plot the behavior of the distance
plt.figure(fig)
plt.plot(distanceAf, 'r', label='GNode A Distances')
plt.plot(distanceBf, 'g', label='GNode B Distances')
plt.plot(distanceCf, 'b', label='GNode C Distances')
plt.plot(np.arange(len(distanceAf)),np.ones([1,len(distanceAf)])[0]*AfAve, 'r.', label='Average GNode A Distance')
plt.plot(np.arange(len(distanceAf)),np.ones([1,len(distanceAf)])[0]*BfAve, 'g.', label='Average GNode B Distance')
plt.plot(np.arange(len(distanceAf)),np.ones([1,len(distanceAf)])[0]*CfAve, 'b.', label='Average GNode C Distance')
plt.plot(np.arange(len(distanceAf)),np.ones([1,len(distanceAf)])[0]*comp_distanceAf, 'r--', label='Actual GNode A Distance')
plt.plot(np.arange(len(distanceAf)),np.ones([1,len(distanceAf)])[0]*comp_distanceBf, 'g--', label='Actual GNode B Distance')
plt.plot(np.arange(len(distanceAf)),np.ones([1,len(distanceAf)])[0]*comp_distanceCf, 'b--', label='Actual GNode C Distance')
plt.plot([], [], ' ', label=' ') # Dummy Plots for Initial Parameters
plt.plot([], [], ' ', label='Parameters:')
plt.plot([], [], ' ', label='n = '+str(n))
plt.plot([], [], ' ', label='$D_{RSSIo} = $'+str(dro))
plt.plot([], [], ' ', label='$RSSI_o = $'+str(roRSSI))
plt.title(dtn + ' 0' + phoneA  + ' Distance Behavior')
plt.xlabel('RSSI Index No.')
plt.ylabel('Distance [Meters]')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1.03)) 
plt.savefig(save_destination + dtn + ' 0' + phoneA + ' DistanceBehavior.jpg', bbox_inches='tight')
plt.close()
fig += 1

# Plot the behavior of the RSSI Kalman and Raw
plt.figure(fig)
plt.plot(rssiA_kalman, 'r', label='GNode A RSSI w/ Kalman')
plt.plot(rssiB_kalman, 'g', label='GNode B RSSI w/ Kalman')
plt.plot(rssiC_kalman, 'b', label='GNode C RSSI w/ Kalman')
plt.plot(rssiA_int, 'r', alpha=0.3, label='GNode A RSSI')
plt.plot(rssiB_int, 'g', alpha=0.3, label='GNode B RSSI')
plt.plot(rssiC_int, 'b', alpha=0.3, label='GNode C RSSI')
rssiAAveK = sum(rssiA_kalman)/len(rssiA_kalman)
rssiBAveK = sum(rssiB_kalman)/len(rssiB_kalman)
rssiCAveK = sum(rssiC_kalman)/len(rssiC_kalman)
plt.plot(np.arange(len(rssiA_kalman)),np.ones([1,len(rssiA_kalman)])[0]*rssiAAveK, 'r.', alpha=1, markersize = 0.8, label='Average GNode A RSSI')
plt.plot(np.arange(len(rssiA_kalman)),np.ones([1,len(rssiA_kalman)])[0]*rssiBAveK, 'g.', alpha=1, markersize = 0.8, label='Average GNode B RSSI')
plt.plot(np.arange(len(rssiA_kalman)),np.ones([1,len(rssiA_kalman)])[0]*rssiCAveK, 'b.', alpha=1, markersize = 0.8, label='Average GNode C RSSI')
plt.plot([], [], ' ', label=' ') # Dummy Plots for Initial Parameters
plt.plot([], [], ' ', label='Parameters:')
plt.plot([], [], ' ', label='n = '+str(n))
plt.plot([], [], ' ', label='$D_{RSSIo} = $'+str(dro))
plt.plot([], [], ' ', label='$RSSI_o = $'+str(roRSSI))
plt.title(dtn + ' 0' + phoneA  + ' RSSI Behavior')
plt.xlabel('RSSI Index No.')
plt.ylabel('dBm')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1.03)) 
plt.savefig(save_destination + dtn + ' 0' + phoneA + ' RSSIBehavior.jpg', bbox_inches='tight')
plt.close()
fig += 1

# Plot the data for trilateration
plt.figure(fig)
plt.scatter(x, y, label='Mobile Node Locations', c='blue', s=20)
plt.scatter(xAct, yAct, marker='*', label='Actual Point', c='darkorange', s=30)
plt.scatter(xg, yg, marker='1', label='GNode Locations', c='black', s=20)
plt.scatter([], [], marker = ' ', label=' ') # Dummy Plots for Initial Parameters
plt.scatter([], [], marker=' ', label='Parameters:')
plt.scatter([], [], marker=' ', label='n = '+str(n))
plt.scatter([], [], marker=' ', label='$D_{RSSIo} = $'+str(dro))
plt.scatter([], [], marker=' ', label='$RSSI_o = $'+str(roRSSI))
plt.scatter([], [], marker=' ', label='Circle Points = '+str(points))
plt.grid(linewidth=1, color="w")
ax = plt.gca()
ax.set_facecolor('gainsboro')
ax.set_axisbelow(True)
plt.title(dtn + ' 0' + phoneA  + ' Raw Trilateration', y=1.05)
plt.xlabel('x-axis [Meters]')
plt.ylabel('y-axis [Meters]')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1.03)) 
plt.savefig(save_destination + dtn + ' 0' + phoneA + ' RawTrilateration.jpg', bbox_inches='tight')
plt.close()
fig += 1

# Raw Trilateration Plot Folium Mapping
latDataOld, longDataOld = cartToGPS(xOld,yOld)
latAveOld, longAveOld = cartToGPS(np.array([xAveOld]), np.array([yAveOld]))
latData, longData = cartToGPS(x,y)
latAve, longAve = cartToGPS(np.array([xAve]), np.array([yAve]))
latAct, longAct = cartToGPS(xAct, yAct)

# Establish Folium Map
m = folium.Map(location=[latAct[0], longAct[0]], zoom_start=20)

# Add New Trilateration
for i in range(len(latData)):
    folium.Circle(
        radius=1,
        location=[latData[i], longData[i]],
        tooltip='New Trilateration',
        popup=str(latData[i])+','+str(longData[i]),
        color='blue',
        fill='True'
    ).add_to(m)

# Add New Trilateration Average
folium.Circle(
    radius=1,
    location=[latAve[0], longAve[0]],
    tooltip='Average Point',
    popup=str(latAve[0])+','+str(longAve[0]),
    color='lightblue',
    fill='True'
).add_to(m)

# Add Actual Point
folium.Marker(
    location=[latAct[0], longAct[0]],
    tooltip='Actual Point',
    popup=str(latAct[0])+','+str(longAct[0]),
    icon=folium.Icon(color='black', icon='star', prefix='fa'),
).add_to(m)

# Add GNode Locations
for i in range(len(latg)):
    folium.Marker(
        location=[latg[i], longg[i]],
        tooltip='GNode Locations',
        popup=str(latg[i])+','+str(longg[i]),
        icon=folium.Icon(color='black', icon='hdd-o', prefix='fa'),
    ).add_to(m)

m.save(save_destination + dtn + ' 0' + phoneA + ' RawTrilaterationMap.html')

# Old vs Improved Trilateration Plot Folium Mapping
# Establish Folium Map
m = folium.Map(location=[latg[0], longg[0]], zoom_start=20)

# Add Old Trilateration
for i in range(len(latDataOld)):
    folium.Circle(
        radius=1,
        location=[latDataOld[i], longDataOld[i]],
        tooltip='Old Trilateration',
        popup=str(latDataOld[i])+','+str(longDataOld[i]),
        color='red',
        fill='True'
    ).add_to(m)

# Add Old Trilateration Average
folium.Circle(
    radius=1,
    location=[latAveOld[0], longAveOld[0]],
    tooltip='Old Trilateration Average',
    popup=str(latAveOld[0])+','+str(longAveOld[0]),
    color='orange',
    fill='True'
).add_to(m)

# Add New Trilateration
for i in range(len(latData)):
    folium.Circle(
        radius=1,
        location=[latData[i], longData[i]],
        tooltip='New Trilateration',
        popup=str(latData[i])+','+str(longData[i]),
        color='blue',
        fill='True'
    ).add_to(m)

# Add New Trilateration Average
folium.Circle(
    radius=1,
    location=[latAve[0], longAve[0]],
    tooltip='New Trilateration Average',
    popup=str(latAve[0])+','+str(longAve[0]),
    color='lightblue',
    fill='True'
).add_to(m)

# Add Actual Point
folium.Marker(
    location=[latAct[0], longAct[0]],
    tooltip='Actual Point',
    popup=str(latAct[0])+','+str(longAct[0]),
    icon=folium.Icon(color='black', icon='star', prefix='fa'),
).add_to(m)

# Add GNode Locations
for i in range(len(latg)):
    folium.Marker(
        location=[latg[i], longg[i]],
        tooltip='GNode Locations',
        popup=str(latg[i])+','+str(longg[i]),
        icon=folium.Icon(color='black', icon='hdd-o', prefix='fa'),
    ).add_to(m)

m.save(save_destination + dtn + ' 0' + phoneA + ' OldVImprovedTrilaterationMap.html') 

# K-Means
print('Performing K-Means...')
# K-means Clustering won't be performed if there is only 1 set of coordinates in the Dataset.
if len(xFilt)<2:
    print("K-means clustering can't be performed due to lack of sample coordinates")
    quit()

# Create numpy array 'data' for K-means containing (xFilt,yFilt) coordinates
data = np.array([[xFilt[0],yFilt[0]]])
for i in range(1,len(xFilt)):
    data = np.append(data,[[xFilt[i],yFilt[i]]], axis=0)

# Mobile Node Duplicate Coordinates Filter for K-means Convergence
data = np.unique(data, axis=0) #Eliminate Duplicates in data

kmeans,inertia,elbow = kmeansOptimize(data)
print('Optimal Number of Clusters is', elbow.knee)
print('K-Means Done!\n')

# K-Means Elbow Plot
plt.figure(fig)
plt.plot(range(1,len(data)), inertia)
plt.plot(elbow.knee, elbow.knee_y, 'ro', label='Optimal Clusters: ' + str(elbow.knee))
plt.plot([], [], ' ', label='@ SoSD: ' + str("{:.4f}".format(elbow.knee_y)))
plt.xlabel('No. of Clusters')
plt.ylabel('Sum of Squared Distances')
plt.title(dtn + ' 0' + phoneA  + ' K-Means Elbow')
plt.legend() 
plt.savefig(save_destination + dtn + ' 0' + phoneA + ' K-MeansElbow.jpg') #Change Directory Accordingly
fig += 1

# K-means Plot
plt.figure(fig)
plt.scatter(data[:,0], data[:,1], label = 'Mobile Node Locations', c=kmeans.labels_, cmap='brg', s=5)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c=list(range(1,elbow.knee+1)), marker='x', label ='Cluster Centers', cmap='brg', s=30)
plt.scatter(xAct, yAct, marker='*', label='Actual Point', c='darkorange', s=30)
plt.scatter(xg, yg, marker='1', label='GNode Locations', c='black', s=30)
plt.scatter([], [], marker = ' ', label=' ') # Dummy Plots for Initial Parameters
plt.scatter([], [], marker=' ', label='Parameters: ')
plt.scatter([], [], marker=' ', label='n = '+ str(n))
plt.scatter([], [], marker=' ', label='$D_{RSSIo} = $'+ str(dro))
plt.scatter([], [], marker=' ', label='$RSSI_o = $'+ str(roRSSI))
plt.scatter([], [], marker=' ', label='Circle Points = '+ str(points))
plt.scatter([], [], marker=' ', label='No. of Clusters  = '+ str(elbow.knee))
plt.grid(linewidth=1, color="w")
ax = plt.gca()
ax.set_facecolor('gainsboro')
ax.set_axisbelow(True)
plt.xlabel('x-axis [Meters]')
plt.ylabel('y-axis [Meters]')
plt.title(dtn + ' 0' + phoneA  + ' K-Means', y=1.05)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1.03)) 
plt.savefig(save_destination + dtn + ' 0' + phoneA + ' K-Means.jpg', bbox_inches='tight') #Change Directory Accordingly
fig += 1

# K-means Plot Folium Mapping
# Cartesian to GPS Coordinate Conversion
latData, longData = cartToGPS(data[:,0],data[:,1])
latCenter, longCenter = cartToGPS(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1])
latAve, longAve = cartToGPS(np.array([xAve]), np.array([yAve]))
latAct, longAct = cartToGPS(xAct, yAct)

# Establish Folium Map
m = folium.Map(location=[latg[0], longg[0]], zoom_start=20)

# Add Mobile Node Locations to Folium Map
for i in range(len(latData)):
    folium.Circle(
        radius=1,
        location=[latData[i], longData[i]],
        tooltip='Mobile Node Locations',
        popup=str(latData[i])+','+str(longData[i]),
        color='red',
        fill='True'
    ).add_to(m)

# Add Cluster Centers
for i in range(len(latCenter)):
    folium.Circle(
        radius=1,
        location=[latCenter[i], longCenter[i]],
        tooltip='Cluster Centers',
        popup=str(latCenter[i])+','+str(longCenter[i]),
        color='blue',
        fill='True'
    ).add_to(m)

# Add Actual Point
folium.Marker(
    location=[latAct[0], longAct[0]],
    tooltip='Actual Point',
    popup=str(latAct[0])+','+str(longAct[0]),
    icon=folium.Icon(color='black', icon='star', prefix='fa'),
).add_to(m)

# Add GNode Locations
for i in range(len(latg)):
    folium.Marker(
        location=[latg[i], longg[i]],
        tooltip='GNode Locations',
        popup=str(latg[i])+','+str(longg[i]),
        icon=folium.Icon(color='black', icon='hdd-o', prefix='fa'),
    ).add_to(m)

# Save HTML Map File
m.save(save_destination + dtn + ' 0' + phoneA + ' K-MeansMap.html')

# DBSCAN
print('Performing DBSCAN...')

dbData = np.array([[xFilt[0],yFilt[0]]])
for i in range(1,len(xFilt)):
    dbData = np.append(dbData,[[xFilt[i],yFilt[i]]], axis=0)

dbscan, nNeighborDistance, dbElbow = dbscanOptimize(dbData, minPts, kNeighbors)
print('Optimal Value for Epsilon is', dbElbow.knee_y)
print('MinPts required for each cluster is', minPts)

print('DBSCAN Done!\n')

# DBSCAN Elbow Plot
plt.figure(fig)
plt.plot(range(0,len(dbData)), nNeighborDistance)
plt.plot(dbElbow.knee, dbElbow.knee_y, 'ro', label='Optimal ε: ' + str("{:.4f}".format(dbElbow.knee_y)))
plt.xlabel('Nearest Neighbor Distance Index No.')
plt.ylabel('Distance from Nearest Neighbor [Meters]')
plt.title(dtn + ' 0' + phoneA  + ' DBSCAN Elbow')
plt.legend() 
plt.savefig(save_destination + dtn + ' 0' + phoneA + ' DBSCANElbow.jpg') #Change Directory Accordingly
fig += 1
    
# DBSCAN Plot
plt.figure(fig)
plt.scatter(dbData[dbscan.labels_>-1,0], dbData[dbscan.labels_>-1,1], label ='Mobile Node Clusters', c=dbscan.labels_[dbscan.labels_>-1], cmap='brg', s=5)
plt.scatter(dbData[dbscan.labels_==-1,0], dbData[dbscan.labels_==-1,1], marker='x', label='Noise', c='darkkhaki', s=15)
plt.scatter(xAct, yAct, marker='*', label='Actual Point', c='darkorange', s=30)
plt.scatter(xg, yg, marker='1', label='GNode Locations', c='black', s=30)
plt.scatter([], [], marker = ' ', label=' ') # Dummy Plots for Initial Parameters
plt.scatter([], [], marker=' ', label='Parameters: ')
plt.scatter([], [], marker=' ', label='n = '+ str(n))
plt.scatter([], [], marker=' ', label='$D_{RSSIo} = $'+ str(dro))
plt.scatter([], [], marker=' ', label='$RSSI_o = $'+ str(roRSSI))
plt.scatter([], [], marker=' ', label='Circle Points = '+ str(points))
plt.scatter([], [], marker=' ', label='ε  = '+ str("{:.4f}".format(dbElbow.knee_y)))
plt.scatter([], [], marker=' ', label='MinPts  = '+ str(minPts))
plt.scatter([], [], marker=' ', label='No. of Clusters  = '+ str(max(dbscan.labels_)+1))
plt.grid(linewidth=1, color="w")
ax = plt.gca()
ax.set_facecolor('gainsboro')
ax.set_axisbelow(True)
plt.xlabel('x-axis [Meters]')
plt.ylabel('y-axis [Meters]')
plt.title(dtn + ' 0' + phoneA  + ' DBSCAN', y=1.05)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1.03)) 
plt.savefig(save_destination + dtn + ' 0' + phoneA + ' DBSCAN.jpg', bbox_inches='tight') #Change Directory Accordingly
fig += 1

# DBSCAN Plot Folium Mapping

# Cartesian to GPS Coordinate Conversion
latData, longData = cartToGPS(dbData[dbscan.labels_>-1,0],dbData[dbscan.labels_>-1,1])
latAve, longAve = cartToGPS(np.array([xAve]), np.array([yAve]))
latAct, longAct = cartToGPS(xAct, yAct)

# Establish Folium Map
m = folium.Map(location=[latg[0], longg[0]], zoom_start=20)

# Add Mobile Node Locations to Folium Map
for i in range(len(latData)):
    folium.Circle(
        radius=1,
        location=[latData[i], longData[i]],
        tooltip='Mobile Node Locations',
        popup=str(latData[i])+','+str(longData[i]),
        color='red',
        fill='True'
    ).add_to(m)

# Add Actual Point
folium.Marker(
    location=[latAct[0], longAct[0]],
    tooltip='Actual Point',
    popup=str(latAct[0])+','+str(longAct[0]),
    icon=folium.Icon(color='black', icon='star', prefix='fa'),
).add_to(m)

# Add GNode Locations
for i in range(len(latg)):
    folium.Marker(
        location=[latg[i], longg[i]],
        tooltip='GNode Locations',
        popup=str(latg[i])+','+str(longg[i]),
        icon=folium.Icon(color='black', icon='hdd-o', prefix='fa'),
    ).add_to(m)

# Save HTML Map File
m.save(save_destination + dtn + ' 0' + phoneA + ' DBSCANMap.html')

# Error Computations
# Computed Position vs. Actual Position
compVact, centVave, compVcent, triImprovement = errorComp(x, y, xOld, yOld, xAct, yAct, kmeans, xAve, yAve, xAveOld, yAveOld, data)
compVactAve = sum(compVact)/len(compVact)
compVactMax = max(compVact)
compVactMin = min(compVact)

# Plot Old vs Improved Trilateration
plt.figure(fig,figsize=(10,5))
plt.scatter(xOld, yOld, label='Old Trilateration', c='red', s=20)
plt.scatter(xAveOld, yAveOld, label='Old Trilateration Average', c='orange', s=20)
plt.scatter(x, y, label='Improved Trilateration', c='blue', s=20)
plt.scatter(xAve, yAve, label='Improved Trilateration Average', c='cyan', s=20)
plt.scatter(xAct, yAct, marker='*', label='Actual Point', c='darkorange', s=30)
plt.scatter(xg, yg, marker='1', label='GNode Locations', c='black', s=30)
plt.scatter([], [], marker = ' ', label=' ') # Dummy Plots for Initial Parameters
plt.scatter([], [], marker=' ', label='Parameters:')
plt.scatter([], [], marker=' ', label='n = '+str(n))
plt.scatter([], [], marker=' ', label='$D_{RSSIo} = $'+str(dro))
plt.scatter([], [], marker=' ', label='$RSSI_o = $'+str(roRSSI))
plt.scatter([], [], marker=' ', label='Circle Points = '+str(points))
plt.scatter([], [], marker = ' ', label=' ')
plt.scatter([], [], marker=' ', label='% Improvement = '+ str("{:.4f}".format(triImprovement)) + "%")
plt.grid(linewidth=1, color="w")
ax = plt.gca()
ax.set_facecolor('gainsboro')
ax.set_axisbelow(True)
plt.title(dtn + ' 0' + phoneA  + ' Old vs Improved Trilateration', y=1.05)
plt.xlabel('x-axis [Meters]')
plt.ylabel('y-axis [Meters]')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1.03))
plt.savefig(save_destination + dtn + ' 0' + phoneA + ' OldVImprovedTrilateration.jpg', bbox_inches='tight')
fig += 1

# Plot the behavior of the error
plt.figure(fig)
plt.plot(compVact, 'r', label='Trilateration Error')
plt.plot(np.arange(len(distanceAf)),np.ones([1,len(distanceAf)])[0]*compVactAve , 'r--', label='Average Error')
# plt.plot(np.arange(len(distanceAf)),np.ones([1,len(distanceAf)])[0]*comp_distanceAf, 'r--', label='Actual GNode A Distance')
plt.plot([], [], ' ', label=' ') # Dummy Plots for Initial Parameters
plt.plot([], [], ' ', label='Parameters:')
plt.plot([], [], ' ', label='n = '+str(n))
plt.plot([], [], ' ', label='$D_{RSSIo} = $'+str(dro))
plt.plot([], [], ' ', label='$RSSI_o = $'+str(roRSSI))
plt.plot([], [], ' ', label=' ') # Dummy Plots for Initial Parameters
plt.plot([], [], ' ', label='Output:')
plt.plot([], [], ' ', label='Average Error = '+str("{:.4f}".format(compVactAve[0])))
plt.plot([], [], ' ', label='Max Error = '+str("{:.4f}".format(compVactMax[0])))
plt.plot([], [], ' ', label='Min Error = '+str("{:.4f}".format(compVactMin[0])))
plt.title(dtn + ' 0' + phoneA  + ' Error Behavior')
plt.xlabel('Datapoint')
plt.ylabel('Distance [Meters]')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1.03)) 
plt.savefig(save_destination + dtn + ' 0' + phoneA + ' ErrorBehavior.jpg', bbox_inches='tight')
plt.close('all')
fig += 1

# # CSV Writing
# print('Saving to CSV...')
# with open(save_destination+'Basic.csv', mode='a') as blogs:
#     blogswrite = csv.writer(blogs, dialect='excel', lineterminator='\n')
#     blogswrite.writerow(['Time',dtn])
#     blogswrite.writerow(['Phone#','0'+phoneA])
#     blogswrite.writerow(['gnodeA',np.append(xg[0],yg[0])])
#     blogswrite.writerow(['gnodeB',np.append(xg[1],yg[1])])
#     blogswrite.writerow(['gnodeC',np.append(xg[2],yg[2])])
#     blogswrite.writerow(['Mean Raw Distances'])
#     blogswrite.writerow(['A','B','C'])
#     blogswrite.writerow([AfAve,BfAve,CfAve])
#     blogswrite.writerow(['Mean Raw X and Y Coordinates','','','',np.append(xAve,yAve)])
#     blogswrite.writerow(['Mean Coordinates with Tolerance Filter','','','',np.append(xFiltAve,yFiltAve)])
#     blogswrite.writerow(['Optimal # of Clusters','',elbow.knee])
#     blogswrite.writerow([''])
#     blogswrite.writerow([''])
    
# with open(save_destination+'DistanceConstants.csv', mode='a') as blogs:
#     blogswrite = csv.writer(blogs, dialect='excel', lineterminator='\n')
#     blogswrite.writerow(['Time',dtn])
#     blogswrite.writerow(['Phone#','0'+phoneA])
#     blogswrite.writerow(['n',n])
#     blogswrite.writerow(['dro',dro])
#     blogswrite.writerow(['RO RSSI',roRSSI])
#     blogswrite.writerow(['Circumference Points',points])
#     blogswrite.writerow([''])
#     blogswrite.writerow([''])
    
# with open(save_destination+'Actual.csv', mode='a') as alogs:
#     alogswrite = csv.writer(alogs, dialect='excel', lineterminator='\n')
#     alogswrite.writerow(['Time',dtn])
#     alogswrite.writerow(['Phone#','0'+phoneA])
#     alogswrite.writerow(['Actual Coordinates','',np.append(xAct,yAct)])
#     alogswrite.writerow(['Actual Computed Distances from Gnodes'])
#     alogswrite.writerow(['A','','B','','C'])
#     alogswrite.writerow([comp_distanceAf,'',comp_distanceBf,'',comp_distanceCf])
#     alogswrite.writerow(['Trilateration Error vs Actual Coordinates'])
#     for i in range(np.shape(compVact)[0]):
#         alogswrite.writerow([compVact[i]])
#     alogswrite.writerow([''])
#     alogswrite.writerow([''])

# with open(save_destination+'Coordinates.csv', mode='a') as clogs:
#     clogswrite = csv.writer(clogs, dialect='excel', lineterminator='\n')
#     clogswrite.writerow(['Time',dtn])
#     clogswrite.writerow(['Phone#','0'+phoneA])
#     clogswrite.writerow(['Raw X and Y Coordinates'])
#     for i in range(np.shape(x)[0]):
#         clogswrite.writerow([np.append(x[i],y[i])])
#     clogswrite.writerow(['-------------------------------'])
#     clogswrite.writerow(['Coordinates with Tolerance Filter'])
#     for i in range(np.shape(xFilt)[0]):
#         clogswrite.writerow([np.append(xFilt[i],yFilt[i])])
#     clogswrite.writerow([''])
#     clogswrite.writerow([''])
    
# with open(save_destination+'Distances.csv', mode='a') as dlogs:
#     dlogswrite = csv.writer(dlogs, dialect='excel', lineterminator='\n')
#     dlogswrite.writerow(['Time',dtn])
#     dlogswrite.writerow(['Phone#','0'+phoneA])
#     dlogswrite.writerow(['Raw Distances'])
#     dlogswrite.writerow(['A','B','C'])
#     for i in range(len(distanceAf)):
#         dlogswrite.writerow([distanceAf[i],distanceBf[i],distanceCf[i]])    
#     dlogswrite.writerow([''])
#     dlogswrite.writerow([''])
    
#     with open(save_destination+'K-Means.csv', mode='a') as klogs:
#         klogswrite = csv.writer(klogs, dialect='excel', lineterminator='\n')
#         klogswrite.writerow(['Time',dtn])
#         klogswrite.writerow(['Phone#','0'+phoneA])
#         klogswrite.writerow(['Inertia'])
#         for i in range(len(inertia)):
#             klogswrite.writerow([inertia[i]]) 
#         klogswrite.writerow(['K-Means Centroid Coordinates'])
#         for i in range(elbow.knee):
#             klogswrite.writerows([[np.append(kmeans.cluster_centers_[i,0],kmeans.cluster_centers_[i,1])]]) 
#         klogswrite.writerow(['K-Means Centroids vs. Mean Coordinates with Tolerance Filter'])
#         klogswrite.writerows([centVave])
#         klogswrite.writerow(['K-Means Centroids vs. Coordinates w/ Tolerance Filter '])
#         for i in range(len(compVcent)):    
#             for j in range (len(compVcent[i])):
#                 klogswrite.writerow([compVcent[i][j]])
#             klogswrite.writerow(['-------------------------------'])
#         klogswrite.writerow([''])
#         klogswrite.writerow([''])

# # Firebase Realtime Database
# print('Uploading to LoRa Rescue Realtime Database...')
# firebase = pyrebase.initialize_app(LoraRescueStorage)
# db = firebase.database()
# dataBasic = {"GNode A":' '.join([str(item) for item in list(np.append(xg[0],yg[0]))]),
#         "GNode B":' '.join([str(item) for item in list(np.append(xg[1],yg[1]))]),
#         "GNode C":' '.join([str(item) for item in list(np.append(xg[2],yg[2]))]),
#         "Distance A Mean":AfAve,"Distance B Mean":BfAve,"Distance C Mean":CfAve,
#         "Mean X and Y Coordinates":' '.join([str(item) for item in list(np.append(xAve,yAve))]),
#         "Mean Filtered X and Y Coordinates":' '.join([str(item) for item in list(np.append(xFiltAve,yFiltAve))]),
#         "Optimal Number of Clusters":int(elbow.knee)}
# dataActual = {"Actual Coordinates":' '.join([str(item).replace("[","").replace("]","") for item in list(np.append(xAct,yAct))]),
#         "Actual Computed Distances from Gnodes (A B C)":str(comp_distanceAf).replace("[","").replace("]","")+" "+str(comp_distanceBf).replace("[","").replace("]","")+" "+str(comp_distanceCf).replace("[","").replace("]",""),
#         "Trilateration Error vs Actual Coordinates":[str(item).replace("[","").replace("]","") for item in compVact]}
# dataCoordinates = {"Raw X":list(x), "Raw Y":list(y),
#         "Filtered X":list(xFilt), "Filtered Y":list(yFilt)}
# dataDistances = {"Distance to GNode A":list(distanceAf),
#         "Distance to GNode B":list(distanceBf),
#         "Distance to GNode C":list(distanceCf)}
# dataDistanceCalc = {"n":n,
#         "dro":dro,
#         "roRSSI":roRSSI,
#         "Circumference Points":points}

# clusterCenterX = list()
# clusterCenterY = list()
# clusterCompVcent = list()
# for i in range(elbow.knee):
#         clusterCenterX.append(''.join([str(item) for item in list(str(kmeans.cluster_centers_[i,0]))]))
#         clusterCenterY.append(''.join([str(item) for item in list(str(kmeans.cluster_centers_[i,1]))]))
# for i in range(len(compVcent)):    
#         for j in range (len(compVcent[i])):
#                 clusterCompVcent.append(compVcent[i][j])

# dataKmeans = {"Intertia":list(inertia),
#         "Centroid X":list(clusterCenterX),
#         "Centroid Y":list(clusterCenterY),
#         "Centroids vs Mean Coordinates w Tolerance Filter":list(centVave),
#         "Centroids vs Coordinates w Tolerance Filter":list(clusterCompVcent)}

# dateAndTime = dtn.split()
# dateNow = dateAndTime[0]
# timeNow = dateAndTime[1].replace("-",":")
# db.child(dateNow).child(timeNow +' 0'+phoneA).child("Basic Raw Information").set(dataBasic)
# db.child(dateNow).child(timeNow +' 0'+phoneA).child("Distance Calculation Constants").set(dataDistanceCalc)
# db.child(dateNow).child(timeNow +' 0'+phoneA).child("Actual Data").set(dataActual)
# db.child(dateNow).child(timeNow +' 0'+phoneA).child("Raw and Filtered Coordinates").set(dataCoordinates)
# db.child(dateNow).child(timeNow +' 0'+phoneA).child("Distances to Gateway Nodes").set(dataDistances)
# db.child(dateNow).child(timeNow +' 0'+phoneA).child("Kmeans Data").set(dataKmeans)

# # Firebase Storage
# print('Uploading to LoRa Rescue Storage...\n')
# firebaseUpload(LoraRescueStorage, 
#     dtn + ' 0' + phoneA + ' FrequencyDistribution.jpg',
#     'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Distance/FrequencyDistribution.jpg')
# firebaseUpload(LoraRescueStorage, 
#     dtn + ' 0' + phoneA + ' DistanceBehavior.jpg',
#     'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Distance/DistanceBehavior.jpg')
# firebaseUpload(LoraRescueStorage, 
#     dtn + ' 0' + phoneA + ' RSSIBehavior.jpg',
#     'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Distance/RSSIBehavior.jpg')
# firebaseUpload(LoraRescueStorage, 
#     dtn + ' 0' + phoneA + ' ErrorBehavior.jpg',
#     'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Trilateration/ErrorBehavior.jpg')
# firebaseUpload(LoraRescueStorage, 
#     dtn + ' 0' + phoneA + ' RawTrilateration.jpg',
#     'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Trilateration/RawTrilateration.jpg')
# firebaseUpload(LoraRescueStorage, 
#     dtn + ' 0' + phoneA + ' RawTrilaterationMap.html',
#     'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Trilateration/RawTrilaterationMap.html') 
# firebaseUpload(LoraRescueStorage, 
#     dtn + ' 0' + phoneA + ' OldVImprovedTrilateration.jpg',
#     'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Trilateration/OldVImprovedTrilateration.jpg')
# firebaseUpload(LoraRescueStorage, 
#     dtn + ' 0' + phoneA + ' OldVImprovedTrilaterationMap.html',
#     'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Trilateration/OldVImprovedTrilaterationMap.html') 
# firebaseUpload(LoraRescueStorage, 
#     dtn + ' 0' + phoneA + ' K-MeansElbow.jpg',
#     'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Clustering/K-MeansElbow.jpg')
# firebaseUpload(LoraRescueStorage, 
#     dtn + ' 0' + phoneA + ' K-Means.jpg',
#     'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Clustering/K-Means.jpg')
# firebaseUpload(LoraRescueStorage, 
#     dtn + ' 0' + phoneA + ' K-MeansMap.html',
#     'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Clustering/K-MeansMap.html')
# firebaseUpload(LoraRescueStorage, 
#     dtn + ' 0' + phoneA + ' DBSCANElbow.jpg',
#     'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Clustering/DBSCANElbow.jpg')
# firebaseUpload(LoraRescueStorage, 
#     dtn + ' 0' + phoneA + ' DBSCAN.jpg',
#     'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Clustering/DBSCAN.jpg')
# firebaseUpload(LoraRescueStorage, 
#     dtn + ' 0' + phoneA + ' DBSCANMap.html',
#     'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Clustering/DBSCANMap.html')
# print("Done!")