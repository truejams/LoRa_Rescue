# LoRa Rescue Code

# Libraries
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
from selenium import webdriver
import serial
import time
from datetime import datetime as dt
import os
import pyrebase
from sklearn.cluster import DBSCAN

# Variable Declaration
###### CHANGE THIS ACCORDINGLY ######
# Benjamin's Directory
# save_destination = "C:\\Users\\Benj\\Desktop\\LoRa_Rescue\\10-23-21_Data\\"
# browser_driver = "C:\\Users\\Benj\\Desktop\\LoRa_Rescue\\chromedriver.exe"
# Ianny's Directory
save_destination = "D:\\Users\\Yani\\Desktop\\LoRa Rescue Data\\"
browser_driver = "D:\\Users\\Yani\\Desktop\\LoRa Rescue Data\\chromedriver.exe"

# Change Current Working Directory in Python
os.chdir(save_destination)

# Arduino Configuration
###### CHANGE THIS ACCORDINGLY ######
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
###### CHANGE THIS ACCORDINGLY ######
startrow = 1
endrow = 58

# RSSI to Distance calculation constants
###### CHANGE THIS ACCORDINGLY ######
n = 5
dro = 1.5
roRSSI = -32

# Trilateration calculation constants
# GNode GPS Coordinates
# Format: A B C
###### CHANGE THIS ACCORDINGLY ######
latg = np.array([14.6648848,14.6648496,14.6648452])
longg = np.array([120.9718980,120.9718835,120.9718860])

# GNode Cartesian Coordinates
# Format: A B C
xg = np.array([0,0,0])
yg = np.array([0,0,0])

# Actual Mobile Node GPS Coordinates
###### CHANGE THIS ACCORDINGLY ######
latAct = np.array([14.6648547])
longAct = np.array([120.9718816])

# Actual Mobile Node Cartesian Coordinates
xAct = np.array([0]) #Target x-coordinate
yAct = np.array([0]) #Target y-coordinate

# Tolerance filter error margin
###### CHANGE THIS ACCORDINGLY ######
errorTolerance = 50

# DBSCAN calculation constants
###### CHANGE THIS ACCORDINGLY ######
# Will be removed when an optimization function is made
epsilon = 50
clusterSamples = 3

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
                print("timeA: " + timeA)
                dtn = timeA
                dtn = dtn.replace(':','-')
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
            start_dt = dt.strptime(timeA[11:19], '%H:%M:%S')
            end_dt = dt.strptime(timeC[11:19], '%H:%M:%S')
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

def importCSV(save_destination, startrow, endrow):
    rawdataread = pd.read_csv(save_destination + 'rawData.csv', header=0)
    #r"" specifies that it is a string. This is the location of the csv to be read
    #header=0 means headers at row 0

    rawdatalim = rawdataread[startrow-1:endrow-1] #limit for which columns to read.

    phone = rawdatalim['Phone'].to_list()[0] #reads 1st column with Phone header
    dtn  = rawdatalim['Time'].to_list()[0] #reads 1st column with Time header
    dtn = dtn.replace(':','-') #replaces : to ; for saving files
    rssiA = rawdatalim['Gateway A'].to_numpy(float) #reads column with Gateway A header and then converts into numpy float array
    rssiB = rawdatalim['Gateway B'].to_numpy(float) #reads column with Gateway B header and then converts into numpy float array
    rssiC = rawdatalim['Gateway C'].to_numpy(float) #reads column with Gateway C header and then converts into numpy float array
    
    return rssiA, rssiB, rssiC, dtn, phone

def rssiToDist(rssiA,rssiB,rssiC,n,dro,roRSSI):
    distA = list()
    distB = list()
    distC = list()
    rssi = [rssiA,rssiB,rssiC]
    for i in range(len(rssi[0])):
        distA.append(pow(10,((roRSSI-int(rssi[0][i]))/(10*n)))*dro)
        distB.append(pow(10,((roRSSI-int(rssi[1][i]))/(10*n)))*dro)
        distC.append(pow(10,((roRSSI-int(rssi[2][i]))/(10*n)))*dro)

    return distA,distB,distC

def rotateGraph(xg, yg, xAct, yAct):
    def getBcoor(z):
        x = z[0]
        y = z[1]

        F = np.empty((2))
        F[0] = (x**2) + (y**2) - (Rab**2)
        F[1] = ((x-xg[2])**2) + (y**2) - (Rbc**2)
        return F
    def getNcoor(z):
        x = z[0]
        y = z[1]

        F = np.empty((2))
        F[0] = (x**2) + (y**2) - (Ran**2)
        F[1] = ((x-xg[2])**2) + (y**2) - (Rnc**2)
        return F
    zGuess = np.array([1,1])
    if yg[2] != 0:
        notFlat = 1
        xg[2] = -np.sqrt((xg[2]**2)+(yg[2]**2))
        yg[2] = 0
        
        z = fsolve(getBcoor,zGuess)
        xg[1] = z[0]
        yg[1] = -z[1]
        z = fsolve(getNcoor,zGuess)
        xAct = z[0]
        yAct = z[1]
    else:
        notFlat = 0

    return xg, yg, xAct, yAct, notFlat

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

def tolFilter(x,y,errorTolerance):
    i = 0
    while i != 60:
        if i == len(y):
            i = 60
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

    #Determine optimal Number of Clusters based on Elbow
    elbow = KneeLocator(range(1,len(data)),inertia, curve='convex', direction='decreasing')

    #Perform K-means with elbow no. of clusters
    kmeans = KMeans(n_clusters=elbow.knee, n_init=5).fit(data)

    return kmeans,inertia,elbow

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

def errorComp(x, y, xAct, yAct, kmeans, xAve, yAve, data):
    compVact = list()
    for i in range(len(x)):
        compVact.append(np.sqrt((x[i]-xAct)**2+(y[i]-yAct)**2))

    #K-means centroid vs. Average Point (dataset average)
    centVave = np.sqrt((kmeans.cluster_centers_[:,0]-xAve)**2+(kmeans.cluster_centers_[:,1]-yAve)**2)

    #Computed Position vs. K-means centroid
    compVcent = np.sqrt([(data[:,0]-kmeans.cluster_centers_[0,0])**2+(data[:,1]-kmeans.cluster_centers_[0,1])**2])
    for i in range(1,len(kmeans.cluster_centers_)):
        distance = np.sqrt([(data[:,0]-kmeans.cluster_centers_[i,0])**2+(data[:,1]-kmeans.cluster_centers_[i,1])**2])
        compVcent = np.append(compVcent,distance,axis=0)

    return compVact, centVave, compVcent

def actualDist(xAct, yAct, xg, yg):
    #Computed distanceAf, Bf, Cf
    comp_distanceAf = list()
    comp_distanceBf = list()
    comp_distanceCf = list()
    comp_distanceAf = np.sqrt(((xAct-xg[0])**2)+((yAct-yg[0])**2))
    comp_distanceBf = np.sqrt(((xAct-xg[1])**2)+((yAct-yg[1])**2))
    comp_distanceCf = np.sqrt(((xAct-xg[2])**2)+((yAct-yg[2])**2))

    return comp_distanceAf, comp_distanceBf, comp_distanceCf

def firebaseUpload(firebaseConfig, localDir, cloudDir):
    # Initialize Firebase Storage
    firebase = pyrebase.initialize_app(firebaseConfig)
    storage = firebase.storage()

    # Upload files to Firebase Storage
    storage.child(cloudDir).put(localDir)

def dbscan(epsilon, clusterSamples, data, fig):
    db = DBSCAN(eps=epsilon, min_samples=clusterSamples).fit(data)
    dbData = data[db.labels_>-1] 
    dbLabels = db.labels_[db.labels_>-1]
    dbGraph = plt.figure(fig)
    plt.scatter(dbData[:,0],dbData[:,1], c=dbLabels, label = 'Mobile Node Clusters', cmap='brg', s=5)
    plt.scatter(xg, yg, marker='1', label='GNode Locations', c='black', s=30)
    plt.scatter(xAve, yAve, marker='^', label='Average Point', c='black', s=30)
    plt.scatter(xAct, yAct, marker='*', label='Actual Point', c='green', s=30)
    plt.grid(linewidth=1, color="w")
    ax = plt.gca()
    ax.set_facecolor('gainsboro')
    ax.set_axisbelow(True)
    plt.xlabel('x-axis [Meters]')
    plt.ylabel('y-axis [Meters]')
    plt.title(dtn + ' 0' + phoneA  + ' DBSCAN', y=1.05)
    plt.legend()
    plt.savefig(save_destination + dtn + ' 0' + phoneA + ' DBSCAN.jpg') #Change Directory Accordingly
    fig += 1

    return fig

# Listen/Read for Data
# Retrieve RSSI data, date and time, and phone number

# Listen to COM port and check for errors
###### CHANGE THIS ACCORDINGLY ######
# rssiA, rssiB, rssiC, dtn, phoneA = listenForData(port,baud)

# Manually retrieve data from rawData.csv
###### CHANGE THIS ACCORDINGLY ######
rssiA, rssiB, rssiC, dtn, phoneA = importCSV(save_destination, startrow, endrow)

# Save RSSI values to Firebase Database
firebase = pyrebase.initialize_app(LoraRescueStorage)
db = firebase.database()
dataRSSI = {"RSSI Gateway A":list(rssiA),
    "RSSI Gateway B":list(rssiB),
    "RSSI Gateway C":list(rssiC)}
dtemp = dtn
db.child(dtemp.replace("-",":")+' '+'0'+phoneA).child("Raw RSSI Values").set(dataRSSI)

# Convert RSSI to Distance
distanceAf, distanceBf, distanceCf = rssiToDist(rssiA,rssiB,rssiC,n,dro,roRSSI)

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
print("Trilaterating Data...\n")
x,y = trilaterate(distanceAf,distanceBf,distanceCf,xg,yg)
xAve,yAve = trilaterate(AfAve,BfAve,CfAve,xg,yg)

# Tolerance Filter
xFilt,yFilt = tolFilter(x,y,errorTolerance)

# Disable Tolerance Filter
###### CHANGE THIS ACCORDINGLY ######
xFilt = x
yFilt = y

# Mean Coordinates after Tolerance Filter
xFiltAve = np.mean(xFilt)
yFiltAve = np.mean(yFilt)

# Compute actual distance of phone node to GNodes
comp_distanceAf, comp_distanceBf, comp_distanceCf = actualDist(xAct, yAct, xg, yg)

# Plot the data frequency of the gateways
fig = 1
plt.figure(fig)
distSeriesA = pd.Series(distanceAf).value_counts().reset_index().sort_values('index').reset_index(drop=True)
distSeriesA.columns = ['Distance','Frequency']
distSeriesA['Distance'] = distSeriesA['Distance'].round()
distSeriesB = pd.Series(distanceBf).value_counts().reset_index().sort_values('index').reset_index(drop=True)
distSeriesB.columns = ['Distance','Frequency']
distSeriesB['Distance'] = distSeriesB['Distance'].round()
distSeriesC = pd.Series(distanceCf).value_counts().reset_index().sort_values('index').reset_index(drop=True)
distSeriesC.columns = ['Distance','Frequency']
distSeriesC['Distance'] = distSeriesC['Distance'].round()
figur, axes = plt.subplots(1,3, figsize=(18, 5))
axes[0].set_title(dtn + ' 0' + phoneA  + ' A FD')
plots = sns.barplot(ax=axes[0],x="Distance", y="Frequency", data=distSeriesA)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.1f'), 
                    (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                    size=9, xytext=(0, 8),
                    textcoords='offset points')

axes[1].set_title(dtn + ' 0' + phoneA  + ' B FD')
plots = sns.barplot(ax=axes[1],x="Distance", y="Frequency", data=distSeriesB)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.1f'), 
                    (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                    size=9, xytext=(0, 8),
                    textcoords='offset points')

axes[2].set_title(dtn + ' 0' + phoneA  + ' C FD')
plots = sns.barplot(ax=axes[2],x="Distance", y="Frequency", data=distSeriesC)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.1f'), 
                    (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                    size=9, xytext=(0, 8),
                    textcoords='offset points')

plt.savefig(save_destination + dtn + ' 0' + phoneA + ' FrequencyDistribution.jpg')
fig += 1

#For the .py file exclusively, the DB graph unexpectedly mixes with the FD graph without plt.close()
plt.close()

# Plot the behavior of the distance
plt.figure(fig)
plt.plot(distanceAf, label='Gateway A Distances')
plt.plot(distanceBf, label='Gateway B Distances')
plt.plot(distanceCf, label='Gateway C Distances')
plt.plot(np.arange(len(distanceAf)),np.ones([1,len(distanceAf)])[0]*comp_distanceAf, label='Actual GNode A Distance')
plt.plot(np.arange(len(distanceAf)),np.ones([1,len(distanceAf)])[0]*comp_distanceBf, label='Actual GNode B Distance')
plt.plot(np.arange(len(distanceAf)),np.ones([1,len(distanceAf)])[0]*comp_distanceCf, label='Actual GNode C Distance')
plt.title(dtn + ' 0' + phoneA  + ' Distance Behavior')
plt.xlabel('Datapoint')
plt.ylabel('Distance [Meters]')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1.03))
plt.savefig(save_destination + dtn + ' 0' + phoneA + ' DistanceBehavior.jpg', bbox_inches='tight')
fig += 1
# Plot the data for trilateration w/o the filters
plt.figure(fig)
plt.scatter(x, y, label='Mobile Node Locations', cmap='brg', s=20)
plt.scatter(xAve, yAve, label='Ave Node Locations', cmap='brg', s=20)
plt.scatter(xg, yg, marker='1', label='GNode Locations', c='black', s=20)
plt.title(dtn + ' 0' + phoneA  + ' RawTrilateration', y=1.05)
plt.xlabel('x-axis [Meters]')
plt.ylabel('y-axis [Meters]')
plt.legend()
plt.savefig(save_destination + dtn + ' 0' + phoneA + ' RawTrilateration.jpg')
fig += 1
# Plot the data for trilateration w/ the filters
plt.figure(fig)
plt.scatter(xFilt, yFilt, label='Mobile Node Locations', cmap='brg', s=20)
plt.scatter(xFiltAve, yFiltAve, label='Ave Node Locations', cmap='brg', s=20)
plt.scatter(xg, yg, marker='1', label='GNode Locations', c='black', s=20)
plt.title(dtn + ' 0' + phoneA  + ' FiltTrilateration', y=1.05)
plt.xlabel('x-axis [Meters]')
plt.ylabel('y-axis [Meters]')
plt.legend()
plt.savefig(save_destination + dtn + ' 0' + phoneA + ' FiltTrilateration.jpg')
fig += 1

# K-Means
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

# Elbow Plot
plt.figure(fig)
plt.plot(range(1,len(data)), inertia)
plt.xlabel('No. of Clusters')
plt.ylabel('Sum of Squared Distances')
plt.title(dtn + ' 0' + phoneA  + ' Elbow Graph')
plt.savefig(save_destination + dtn + ' 0' + phoneA + ' Elbow.jpg') #Change Directory Accordingly
fig += 1

# K-means Plot
plt.figure(fig)
plt.scatter(data[:,0],data[:,1], c=kmeans.labels_, label = 'Mobile Node Locations', cmap='brg', s=5)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c=list(range(1,elbow.knee+1)), marker = 'x', label = 'Cluster Centers', cmap='brg', s=30)
plt.scatter(xg, yg, marker='1', label='GNode Locations', c='black', s=30)
plt.scatter(xAve, yAve, marker='^', label='Average Point', c='black', s=30)
plt.scatter(xAct, yAct, marker='*', label='Actual Point', c='green', s=30)
plt.grid(linewidth=1, color="w")
ax = plt.gca()
ax.set_facecolor('gainsboro')
ax.set_axisbelow(True)
plt.xlabel('x-axis [Meters]')
plt.ylabel('y-axis [Meters]')
plt.title(dtn + ' 0' + phoneA  + ' K-Means', y=1.05)
plt.legend()
plt.savefig(save_destination + dtn + ' 0' + phoneA + ' K-Means.jpg') #Change Directory Accordingly
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

# Add GNode Locations
for i in range(len(latg)):
    folium.Marker(
        location=[latg[i], longg[i]],
        tooltip='GNode Locations',
        popup=str(latg[i])+','+str(longg[i]),
        icon=folium.Icon(color='black', icon='hdd-o', prefix='fa'),
    ).add_to(m)

# Add Average Points
folium.Circle(
    radius=1,
    location=[latAve[0], longAve[0]],
    tooltip='Average Point',
    popup=str(latAve[0])+','+str(latAve[0]),
    color='black',
    fill='True'
).add_to(m)

# Save HTML Map File
m.save(save_destination + dtn + ' 0' + phoneA + ' FoliumMapping.html')

# Screenshot HTML Map File to get JPG File
options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome(browser_driver, options=options)
driver.get(save_destination + dtn + ' 0' + phoneA + ' FoliumMapping.html')
time.sleep(5) # Delay to accomodate browser snapshotting, change accordingly
driver.save_screenshot(save_destination + dtn + ' 0' + phoneA + ' FoliumMapping.png')
driver.close()

# DBSCAN
# Create numpy array 'dataDB' for DBSCAN containing (x,y) coordinates
dataDB = np.array([[x[0],y[0]]])
for i in range(1,len(xFilt)):
    dataDB = np.append(dataDB,[[x[i],y[i]]], axis=0)

fig = dbscan(epsilon, clusterSamples, dataDB, fig)

# Error Computations
# Computed Position vs. Actual Position
compVact, centVave, compVcent = errorComp(x, y, xAct, yAct, kmeans, xAve, yAve, data)

# CSV Writing
with open(save_destination+'Basic.csv', mode='a') as blogs:
    blogswrite = csv.writer(blogs, dialect='excel', lineterminator='\n')
    blogswrite.writerow(['Time',dtn])
    blogswrite.writerow(['Phone#','0'+phoneA])
    blogswrite.writerow(['gnodeA',np.append(xg[0],yg[0])])
    blogswrite.writerow(['gnodeB',np.append(xg[1],yg[1])])
    blogswrite.writerow(['gnodeC',np.append(xg[2],yg[2])])
    blogswrite.writerow(['Mean Raw Distances'])
    blogswrite.writerow(['A','B','C'])
    blogswrite.writerow([AfAve,BfAve,CfAve])
    blogswrite.writerow(['Mean Raw X and Y Coordinates','','','',np.append(xAve,yAve)])
    blogswrite.writerow(['Mean Coordinates with Tolerance Filter','','','',np.append(xFiltAve,yFiltAve)])
    blogswrite.writerow(['Optimal # of Clusters','',elbow.knee])
    blogswrite.writerow([''])
    blogswrite.writerow([''])
    
with open(save_destination+'Actual.csv', mode='a') as alogs:
    alogswrite = csv.writer(alogs, dialect='excel', lineterminator='\n')
    alogswrite.writerow(['Time',dtn])
    alogswrite.writerow(['Phone#','0'+phoneA])
    alogswrite.writerow(['Actual Coordinates','',np.append(xAct,yAct)])
    alogswrite.writerow(['Actual Computed Distances from Gnodes'])
    alogswrite.writerow(['A','','B','','C'])
    alogswrite.writerow([comp_distanceAf,'',comp_distanceBf,'',comp_distanceCf])
    alogswrite.writerow(['Actual Position vs. Raw X and Y Coordinates'])
    for i in range(np.shape(compVact)[0]):
        alogswrite.writerow([compVact[i]])
    alogswrite.writerow([''])
    alogswrite.writerow([''])

with open(save_destination+'Coordinates.csv', mode='a') as clogs:
    clogswrite = csv.writer(clogs, dialect='excel', lineterminator='\n')
    clogswrite.writerow(['Time',dtn])
    clogswrite.writerow(['Phone#','0'+phoneA])
    clogswrite.writerow(['Raw X and Y Coordinates'])
    for i in range(np.shape(x)[0]):
        clogswrite.writerow([np.append(x[i],y[i])])
    clogswrite.writerow(['-------------------------------'])
    clogswrite.writerow(['Coordinates with Tolerance Filter'])
    for i in range(np.shape(xFilt)[0]):
        clogswrite.writerow([np.append(xFilt[i],yFilt[i])])
    clogswrite.writerow([''])
    clogswrite.writerow([''])
    
with open(save_destination+'Distances.csv', mode='a') as dlogs:
    dlogswrite = csv.writer(dlogs, dialect='excel', lineterminator='\n')
    dlogswrite.writerow(['Time',dtn])
    dlogswrite.writerow(['Phone#','0'+phoneA])
    dlogswrite.writerow(['Raw Distances'])
    dlogswrite.writerow(['A','B','C'])
    for i in range(len(distanceAf)):
        dlogswrite.writerow([distanceAf[i],distanceBf[i],distanceCf[i]])    
    dlogswrite.writerow([''])
    dlogswrite.writerow([''])
    
    with open(save_destination+'K-Means.csv', mode='a') as klogs:
        klogswrite = csv.writer(klogs, dialect='excel', lineterminator='\n')
        klogswrite.writerow(['Time',dtn])
        klogswrite.writerow(['Phone#','0'+phoneA])
        klogswrite.writerow(['Inertia'])
        for i in range(len(inertia)):
            klogswrite.writerow([inertia[i]]) 
        klogswrite.writerow(['K-Means Centroid Coordinates'])
        for i in range(elbow.knee):
            klogswrite.writerows([[np.append(kmeans.cluster_centers_[i,0],kmeans.cluster_centers_[i,1])]]) 
        klogswrite.writerow(['K-Means Centroids vs. Mean Coordinates with Tolerance Filter'])
        klogswrite.writerows([centVave])
        klogswrite.writerow(['K-Means Centroids vs. Coordinates w/ Tolerance Filter '])
        for i in range(len(compVcent)):    
            for j in range (len(compVcent[i])):
                klogswrite.writerow([compVcent[i][j]])
            klogswrite.writerow(['-------------------------------'])
        klogswrite.writerow([''])
        klogswrite.writerow([''])

# Firebase Realtime Database
firebase = pyrebase.initialize_app(LoraRescueStorage)
db = firebase.database()
dataBasic = {"GNode A":' '.join([str(item) for item in list(np.append(xg[0],yg[0]))]),
        "GNode B":' '.join([str(item) for item in list(np.append(xg[1],yg[1]))]),
        "GNode C":' '.join([str(item) for item in list(np.append(xg[2],yg[2]))]),
        "Distance A Mean":AfAve,"Distance B Mean":BfAve,"Distance C Mean":CfAve,
        "Mean X and Y Coordinates":' '.join([str(item) for item in list(np.append(xAve,yAve))]),
        "Mean Filtered X and Y Coordinates":' '.join([str(item) for item in list(np.append(xFiltAve,yFiltAve))]),
        "Optimal Number of Clusters":int(elbow.knee)}
dataActual = {"Actual Coordinates":' '.join([str(item).replace("[","").replace("]","") for item in list(np.append(xAct,yAct))]),
        "Actual Computed Distances from Gnodes (A B C)":str(comp_distanceAf).replace("[","").replace("]","")+" "+str(comp_distanceBf).replace("[","").replace("]","")+" "+str(comp_distanceCf).replace("[","").replace("]",""),
        "Actual Position VS Raw X and Y Coordinates":[str(item).replace("[","").replace("]","") for item in compVact]}
dataCoordinates = {"Raw X":list(x), "Raw Y":list(y),
        "Filtered X":list(xFilt), "Filtered Y":list(yFilt)}
dataDistances = {"Distance to GNode A":list(distanceAf),
        "Distance to GNode B":list(distanceBf),
        "Distance to GNode C":list(distanceCf)}

clusterCenterX = list()
clusterCenterY = list()
clusterCompVcent = list()
for i in range(elbow.knee):
        clusterCenterX.append(''.join([str(item) for item in list(str(kmeans.cluster_centers_[i,0]))]))
        clusterCenterY.append(''.join([str(item) for item in list(str(kmeans.cluster_centers_[i,1]))]))
for i in range(len(compVcent)):    
        for j in range (len(compVcent[i])):
                clusterCompVcent.append(compVcent[i][j])

dataKmeans = {"Intertia":list(inertia),
        "Centroid X":list(clusterCenterX),
        "Centroid Y":list(clusterCenterY),
        "Centroids vs Mean Coordinates w Tolerance Filter":list(centVave),
        "Centroids vs Coordinates w Tolerance Filter":list(clusterCompVcent)}

db.child(dtn.replace("-",":")+' '+'0'+phoneA).child("Basic Raw Information").set(dataBasic)
db.child(dtn.replace("-",":")+' '+'0'+phoneA).child("Actual Data").set(dataActual)
db.child(dtn.replace("-",":")+' '+'0'+phoneA).child("Raw and Filtered Coordinates").set(dataCoordinates)
db.child(dtn.replace("-",":")+' '+'0'+phoneA).child("Distances to Gateway Nodes").set(dataDistances)
db.child(dtn.replace("-",":")+' '+'0'+phoneA).child("Kmeans Data").set(dataKmeans)

# Firebase Realtime Storage
firebaseUpload(LoraRescueStorage, 
    dtn + ' 0' + phoneA + ' FrequencyDistribution.jpg',
    'LoRa Rescue Data/' + dtn + ' 0' + phoneA + ' FrequencyDistribution.jpg')
firebaseUpload(LoraRescueStorage, 
    dtn + ' 0' + phoneA + ' DistanceBehavior.jpg',
    'LoRa Rescue Data/' + dtn + ' 0' + phoneA + ' DistanceBehavior.jpg')
firebaseUpload(LoraRescueStorage, 
    dtn + ' 0' + phoneA + ' RawTrilateration.jpg',
    'LoRa Rescue Data/' + dtn + ' 0' + phoneA + ' RawTrilateration.jpg')    
firebaseUpload(LoraRescueStorage, 
    dtn + ' 0' + phoneA + ' FiltTrilateration.jpg',
    'LoRa Rescue Data/' + dtn + ' 0' + phoneA + ' FiltTrilateration.jpg')
firebaseUpload(LoraRescueStorage, 
    dtn + ' 0' + phoneA + ' Elbow.jpg',
    'LoRa Rescue Data/' + dtn + ' 0' + phoneA + ' Elbow.jpg')
firebaseUpload(LoraRescueStorage, 
    dtn + ' 0' + phoneA + ' K-Means.jpg',
    'LoRa Rescue Data/' + dtn + ' 0' + phoneA + ' K-Means.jpg')
firebaseUpload(LoraRescueStorage, 
    dtn + ' 0' + phoneA + ' FoliumMapping.png',
    'LoRa Rescue Data/' + dtn + ' 0' + phoneA + ' FoliumMapping.png')
firebaseUpload(LoraRescueStorage, 
    dtn + ' 0' + phoneA + ' FoliumMapping.html',
    'LoRa Rescue Data/' + dtn + ' 0' + phoneA + ' FoliumMapping.html')
firebaseUpload(LoraRescueStorage, 
    dtn + ' 0' + phoneA + ' DBSCAN.jpg',
    'LoRa Rescue Data/' + dtn + ' 0' + phoneA + ' DBSCAN.jpg')