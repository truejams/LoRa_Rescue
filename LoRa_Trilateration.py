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
# save_destination = "C:\\LoRa_Rescue\\11-14-21_KalmanTests\\"
# Ianny's Directory
save_destination = "D:\\Users\\Yani\\Desktop\\LoRa Rescue Data\\"
# Greg's Directory
# save_destination = "C:\\LoRa_Rescue\\"

# Change Current Working Directory in Python
os.chdir(save_destination)

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

# Function Declarations
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
        dist.append(pow(10,((roRSSI-int(rssi[i]))/(10*n)))*dro)

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

# Import data from Realtime Database
rssiA, rssiB, rssiC, dtn, phoneA, latg, longg, latAct, longAct =  importDatabase("2021-11-13", "15:41:48 09976500641")

################### RSSI Kalman ######################
rssiA_int = [int(i) for i in rssiA]
rssiB_int = [int(i) for i in rssiB]
rssiC_int = [int(i) for i in rssiC]

rssiA_kalman = kalman_filter(rssiA_int, A=1, H=1, Q=0.005, R=1)
rssiB_kalman = kalman_filter(rssiB_int, A=1, H=1, Q=0.005, R=1)
rssiC_kalman = kalman_filter(rssiC_int, A=1, H=1, Q=0.005, R=1)

# Convert Kalman Filter RSSI to Distance
distanceAf = rssiToDist(rssiA_kalman,nA,dro,roRSSI)
distanceBf = rssiToDist(rssiB_kalman,nB,dro,roRSSI)
distanceCf = rssiToDist(rssiC_kalman,nC,dro,roRSSI)

# Convert GPS Coordinates to Cartesian Coordinates
xg,yg = GPSToCart(latg,longg)
xAct,yAct = GPSToCart(latAct,longAct)

# Trilateration Setup
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

# Old Trilateration
xOld,yOld = trilaterate(distanceAf,distanceBf,distanceCf,xg,yg)
xAveOld,yAveOld = trilaterate(AfAve,BfAve,CfAve,xg,yg)

# New Trilateration
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

# Percent difference/improvement from old to new trilateration
oldtriVact = distanceFormula(xOld,yOld,xAct,yAct)
oldtriVact = np.mean(oldtriVact)
print("oldtriVact:", oldtriVact)
newtriVact = distanceFormula(np.array([x]),np.array([y]),xAct,yAct)
newtriVact = np.mean(newtriVact)
print("newtriVact:", newtriVact)
triImprovement = abs((newtriVact - oldtriVact) / (oldtriVact))*100
print("The percent improvement is", str(triImprovement) + "%")

# Plotting New and Old Trilateration Graphs
fig = 1
plt.figure(fig,figsize=(10,5))
plt.scatter(xOld, yOld, label='Old Trilateration', c='red', s=20)
plt.scatter(xAveOld, yAveOld, label='Old Trilateration Average', c='orange', s=20)
plt.scatter(x, y, label='New Trilateration', c='blue', s=20)
plt.scatter(xAve, yAve, label='New Trilateration Average', c='cyan', s=20)
plt.scatter(xAct, yAct, marker='*', label='Actual Point', c='darkorange', s=30)
plt.scatter(xg, yg, marker='1', label='GNode Locations', c='black', s=30)
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
plt.title(dtn + ' 0' + phoneA  + ' Old vs New Trilateration', y=1.05)
plt.xlabel('x-axis [Meters]')
plt.ylabel('y-axis [Meters]')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1.03))
# plt.tight_layout()
# plt.show(block='False')
plt.savefig(save_destination + dtn + ' 0' + phoneA + ' Old vs New Trilateration.jpg', bbox_inches='tight') 
fig += 1

# Mapping New and Old Trilateration
latDataOld, longDataOld = cartToGPS(xOld,yOld)
latAveOld, longAveOld = cartToGPS(np.array([xAveOld]), np.array([yAveOld]))
latData, longData = cartToGPS(x,y)
latAve, longAve = cartToGPS(np.array([xAve]), np.array([yAve]))
latAct, longAct = cartToGPS(xAct, yAct)

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
    popup=str(latAct[0])+','+str(latAct[0]),
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

m.save(save_destination + dtn + ' 0' + phoneA + ' Old vs New Trilateration.html') 