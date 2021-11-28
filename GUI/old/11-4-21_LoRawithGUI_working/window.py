#LoRa Rescue with Tkinter GUI v3

# Import Libraries
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from numpy import save

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

#Default GUI input values

n_default = 2.8
points_default = 100
com_default = "COM3"
baud_default = 115200
firebase_read_date_default = "2021-10-30"
firebase_read_time_default = "14:46:14"
firebase_read_phone_default = "09976800632"
save_destination_default = "Please choose save destination"

#Set .py location as active directory (NEEDED for GUI)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def btn_clicked():
    print("Button Clicked")

def open_save_dir(): #Function for the Save Destination Button
    global save_destination
    save_destination = filedialog.askdirectory(title="Choose save destination")
    entry8.delete(0, END)
    entry8.insert(0, save_destination)
    
    save_destination = save_destination + '/'
    print("Save destination set to: " + save_destination)
    return

# Firebase Web App Configuration
LoraRescueStorage = {'apiKey': "AIzaSyAN2jdAfGBhbPz446Lho_Jmu2eysU6Hvqw",
    'authDomain': "lora-rescue.firebaseapp.com",
    'databaseURL': "https://lora-rescue-default-rtdb.asia-southeast1.firebasedatabase.app",
    'projectId': "lora-rescue",
    'storageBucket': "lora-rescue.appspot.com",
    'messagingSenderId': "295122276311",
    'appId': "1:295122276311:web:68ce5d4d4cd6763103c592",
    'measurementId': "G-MCPTP8HPLK"}

def btn_importDatabase():
    global rssiA, rssiB, rssiC, dtn, phoneA, latg, longg, latAct, longAct
    global n, dro, roRSSI
    n = float(entry9.get())
    dro = 1.5
    roRSSI = -32
    date = entry15.get()
    time = entry16.get()
    phone = entry17.get()
    rssiA, rssiB, rssiC, dtn, phoneA, latg, longg, latAct, longAct = importDatabase(date, time, phone)
    runCode(rssiA, rssiB, rssiC, n, dro, roRSSI, latg, longg, latAct, longAct)

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

def importDatabase(date, time, phone):
    phoneTime = time + " " + phone
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
    print(latg)
    print(longg)
    return rssiA, rssiB, rssiC, dtn, phone, latg, longg, latAct, longAct

def rssiToDist(rssiA,rssiB,rssiC,n,dro,roRSSI):
    global distA, distB, distC
    distA = list()
    distB = list()
    distC = list()
    rssi = [rssiA,rssiB,rssiC]
    for i in range(len(rssi[0])):
        distA.append(pow(10,((roRSSI-int(rssi[0][i]))/(10*n)))*dro)
        distB.append(pow(10,((roRSSI-int(rssi[1][i]))/(10*n)))*dro)
        distC.append(pow(10,((roRSSI-int(rssi[2][i]))/(10*n)))*dro)

    return distA,distB,distC

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
                    dist[i] = ((xCirc[i][j]-xCirc[i+1][k])**2)+((yCirc[i][j]-yCirc[i+1][k])**2)
                    if dist[i] < deltaDist[i]:
                        deltaDist[i] = dist[i]
                        x[i] = (xCirc[i][j]+xCirc[i+1][k])/2
                        y[i] = (yCirc[i][j]+yCirc[i+1][k])/2
                elif i == 2:
                    dist[i] = ((xCirc[i][j]-xCirc[0][k])**2)+((yCirc[i][j]-yCirc[0][k])**2)
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

def firebaseUpload(firebaseConfig, localDir, cloudDir):
    # Initialize Firebase Storage
    firebase = pyrebase.initialize_app(firebaseConfig)
    storage = firebase.storage()

    # Upload files to Firebase Storage
    storage.child(cloudDir).put(localDir)

def dbscan(epsilon, minPts, data, fig):
    db = DBSCAN(eps=epsilon, min_samples=minPts).fit(data)
    dbData = data[db.labels_>-1] 
    dbLabels = db.labels_[db.labels_>-1]
    dbGraph = plt.figure(fig)
    plt.scatter(dbData[:,0],dbData[:,1], c=dbLabels, label = 'Mobile Node Clusters', cmap='brg', s=5)
    plt.scatter(xg, yg, marker='1', label='GNode Locations', c='black', s=30)
    plt.scatter(xAve, yAve, marker='^', label='Average Point', c='black', s=30)
    plt.scatter(xAct, yAct, marker='*', label='Actual Point', c='green', s=30)
    plt.scatter([], [], marker = ' ', label=' ') # Dummy Plots for Initial Parameters
    plt.scatter([], [], marker=' ', label='Parameters: ')
    plt.scatter([], [], marker=' ', label='n = '+str(n))
    plt.scatter([], [], marker=' ', label='$D_{RSSIo} = $'+str(dro))
    plt.scatter([], [], marker=' ', label='$RSSI_o = $'+str(roRSSI))
    plt.scatter([], [], marker=' ', label='Circle Points = '+str(points))
    plt.scatter([], [], marker=' ', label='ε  = '+str(epsilon))
    plt.scatter([], [], marker=' ', label='MinPts  = '+str(minPts))
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

    return fig

#Code for everything after getting data

def runCode(rssiA, rssiB, rssiC, n, dro, roRSSI, latg, longg, latAct, longAct):
    global points, xAve, yAve, xg, yg, xAct, yAct
    points = int(entry10.get())

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

    print("Trilaterating Data...")
    # x,y = trilaterate(distanceAf,distanceBf,distanceCf,xg,yg)
    # xAve,yAve = trilaterate(AfAve,BfAve,CfAve,xg,yg)
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

    # Tolerance Filter
    errorTolerance = 50
    xFilt,yFilt = tolFilter(x,y,errorTolerance)

    # Disable Tolerance Filter
    ################## CHANGE THIS ACCORDINGLY ##################  
    xFilt = x
    yFilt = y

    # Mean Coordinates after Tolerance Filter
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
    distSeriesA.columns = ['Distance','Frequency']
    distSeriesA['Distance'] = distSeriesA['Distance'].round()
    distSeriesB = pd.Series(distanceBf).value_counts().reset_index().sort_values('index').reset_index(drop=True)
    distSeriesB.columns = ['Distance','Frequency']
    distSeriesB['Distance'] = distSeriesB['Distance'].round()
    distSeriesC = pd.Series(distanceCf).value_counts().reset_index().sort_values('index').reset_index(drop=True)
    distSeriesC.columns = ['Distance','Frequency']
    distSeriesC['Distance'] = distSeriesC['Distance'].round()
    figur, axes = plt.subplots(1,3, figsize=(18, 5))
    axes[0].set_title(dtn + ' 0' + phoneA  + ' GNode A FD')
    plots = sns.barplot(ax=axes[0],x="Distance", y="Frequency", data=distSeriesA)
    for bar in plots.patches:
        plots.annotate(format(bar.get_height(), '.1f'), 
                        (bar.get_x() + bar.get_width() / 2, 
                        bar.get_height()), ha='center', va='center',
                        size=9, xytext=(0, 8),
                        textcoords='offset points')

    axes[1].set_title(dtn + ' 0' + phoneA  + ' GNode B FD')
    plots = sns.barplot(ax=axes[1],x="Distance", y="Frequency", data=distSeriesB)
    for bar in plots.patches:
        plots.annotate(format(bar.get_height(), '.1f'), 
                        (bar.get_x() + bar.get_width() / 2, 
                        bar.get_height()), ha='center', va='center',
                        size=9, xytext=(0, 8),
                        textcoords='offset points')

    axes[2].set_title(dtn + ' 0' + phoneA  + ' GNode C FD')
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
    plt.plot(distanceAf, 'r', label='GNode A Distances')
    plt.plot(distanceBf, 'g', label='GNode B Distances')
    plt.plot(distanceCf, 'b', label='GNode C Distances')
    plt.plot(np.arange(len(distanceAf)),np.ones([1,len(distanceAf)])[0]*comp_distanceAf, 'r--', label='Actual GNode A Distance')
    plt.plot(np.arange(len(distanceAf)),np.ones([1,len(distanceAf)])[0]*comp_distanceBf, 'g--', label='Actual GNode B Distance')
    plt.plot(np.arange(len(distanceAf)),np.ones([1,len(distanceAf)])[0]*comp_distanceCf, 'b--', label='Actual GNode C Distance')
    plt.plot([], [], ' ', label=' ') # Dummy Plots for Initial Parameters
    plt.plot([], [], ' ', label='Parameters:')
    plt.plot([], [], ' ', label='n = '+str(n))
    plt.plot([], [], ' ', label='$D_{RSSIo} = $'+str(dro))
    plt.plot([], [], ' ', label='$RSSI_o = $'+str(roRSSI))
    plt.title(dtn + ' 0' + phoneA  + ' Distance Behavior')
    plt.xlabel('Datapoint')
    plt.ylabel('Distance [Meters]')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1.03)) 
    plt.savefig(save_destination + dtn + ' 0' + phoneA + ' DistanceBehavior.jpg', bbox_inches='tight')
    fig += 1

    # Plot the data for trilateration w/o the filters
    plt.figure(fig)
    plt.scatter(x, y, label='Mobile Node Locations', cmap='brg', s=20)
    plt.scatter(xAve, yAve, label='Average Mobile Node Locations', cmap='brg', s=20)
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
    fig += 1

    # Plot the data for trilateration w/ the filters
    plt.figure(fig)
    plt.scatter(xFilt, yFilt, label='Mobile Node Locations', cmap='brg', s=20)
    plt.scatter(xFiltAve, yFiltAve, label='Average Mobile Node Locations', cmap='brg', s=20)
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
    plt.title(dtn + ' 0' + phoneA  + ' Filtered Trilateration', y=1.05)
    plt.xlabel('x-axis [Meters]')
    plt.ylabel('y-axis [Meters]')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1.03)) 
    plt.savefig(save_destination + dtn + ' 0' + phoneA + ' FiltTrilateration.jpg', bbox_inches='tight')
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
    plt.plot([elbow.knee], inertia[elbow.knee-1], 'ro', label='Optimal Clusters: ' + str(elbow.knee))
    plt.plot([], [], ' ', label='@ SoSD: ' + str("{:.4f}".format(inertia[elbow.knee-1])))
    plt.xlabel('No. of Clusters')
    plt.ylabel('Sum of Squared Distances')
    plt.title(dtn + ' 0' + phoneA  + ' K-Means Elbow Graph')
    plt.legend() 
    plt.savefig(save_destination + dtn + ' 0' + phoneA + ' K-MeansElbow.jpg') #Change Directory Accordingly
    fig += 1

    # K-means Plot
    plt.figure(fig)
    plt.scatter(data[:,0],data[:,1], c=kmeans.labels_, label = 'Mobile Node Locations', cmap='brg', s=5)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c=list(range(1,elbow.knee+1)), marker = 'x', label = 'Cluster Centers', cmap='brg', s=30)
    plt.scatter(xg, yg, marker='1', label='GNode Locations', c='black', s=30)
    plt.scatter(xAve, yAve, marker='^', label='Average Point', c='black', s=30)
    plt.scatter(xAct, yAct, marker='*', label='Actual Point', c='green', s=30)
    plt.scatter([], [], marker = ' ', label=' ') # Dummy Plots for Initial Parameters
    plt.scatter([], [], marker=' ', label='Parameters: ')
    plt.scatter([], [], marker=' ', label='n = '+str(n))
    plt.scatter([], [], marker=' ', label='$D_{RSSIo} = $'+str(dro))
    plt.scatter([], [], marker=' ', label='$RSSI_o = $'+str(roRSSI))
    plt.scatter([], [], marker=' ', label='Circle Points = '+str(points))
    plt.scatter([], [], marker=' ', label='No. of Clusters  = '+str(elbow.knee))
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

    # DBSCAN

    # DBSCAN calculation constants
    ################## CHANGE THIS ACCORDINGLY ##################  
    # Will be removed when an optimization function is made
    epsilon = 50
    minPts = 3

    # Create numpy array 'dataDB' for DBSCAN containing (x,y) coordinates
    dataDB = np.array([[x[0],y[0]]])
    for i in range(1,len(xFilt)):
        dataDB = np.append(dataDB,[[x[i],y[i]]], axis=0)
    fig = dbscan(epsilon, minPts, dataDB, fig)

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
        
    with open(save_destination+'DistanceConstants.csv', mode='a') as blogs:
        blogswrite = csv.writer(blogs, dialect='excel', lineterminator='\n')
        blogswrite.writerow(['Time',dtn])
        blogswrite.writerow(['Phone#','0'+phoneA])
        blogswrite.writerow(['n',n])
        blogswrite.writerow(['dro',dro])
        blogswrite.writerow(['RO RSSI',roRSSI])
        blogswrite.writerow(['Circumference Points',points])
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
    dataDistanceCalc = {"n":n,
            "dro":dro,
            "roRSSI":roRSSI,
            "Circumference Points":points}

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

    dateAndTime = dtn.split()
    dateNow = dateAndTime[0]
    timeNow = dateAndTime[1].replace("-",":")
    db.child(dateNow).child(timeNow +' 0'+phoneA).child("Basic Raw Information").set(dataBasic)
    db.child(dateNow).child(timeNow +' 0'+phoneA).child("Distance Calculation Constants").set(dataDistanceCalc)
    db.child(dateNow).child(timeNow +' 0'+phoneA).child("Actual Data").set(dataActual)
    db.child(dateNow).child(timeNow +' 0'+phoneA).child("Raw and Filtered Coordinates").set(dataCoordinates)
    db.child(dateNow).child(timeNow +' 0'+phoneA).child("Distances to Gateway Nodes").set(dataDistances)
    db.child(dateNow).child(timeNow +' 0'+phoneA).child("Kmeans Data").set(dataKmeans)

    # Firebase Storage
    firebaseUpload(LoraRescueStorage, 
        save_destination + dtn + ' 0' + phoneA + ' FrequencyDistribution.jpg',
        'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Distance/FrequencyDistribution.jpg')
    firebaseUpload(LoraRescueStorage, 
        save_destination + dtn + ' 0' + phoneA + ' DistanceBehavior.jpg',
        'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Distance/DistanceBehavior.jpg')
    firebaseUpload(LoraRescueStorage, 
        save_destination + dtn + ' 0' + phoneA + ' RawTrilateration.jpg',
        'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Trilateration/RawTrilateration.jpg')    
    firebaseUpload(LoraRescueStorage, 
        save_destination + dtn + ' 0' + phoneA + ' FiltTrilateration.jpg',
        'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Trilateration/FiltTrilateration.jpg')
    firebaseUpload(LoraRescueStorage, 
        save_destination + dtn + ' 0' + phoneA + ' K-MeansElbow.jpg',
        'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Clustering/K-MeansElbow.jpg')
    firebaseUpload(LoraRescueStorage, 
        save_destination + dtn + ' 0' + phoneA + ' K-Means.jpg',
        'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Clustering/K-Means.jpg')
    firebaseUpload(LoraRescueStorage, 
        save_destination + dtn + ' 0' + phoneA + ' FoliumMapping.html',
        'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Trilateration/FoliumMapping.html')
    firebaseUpload(LoraRescueStorage, 
        save_destination + dtn + ' 0' + phoneA + ' DBSCAN.jpg',
        'LoRa Rescue Data/' + dtn[0:10] + '/' + dtn[11:19].replace("-",":") + ' 0' + phoneA + '/Clustering/DBSCAN.jpg')
        
    print("Done!")

window = Tk()

window.geometry("1200x800")
window.configure(bg = "#ffffff")
canvas = Canvas(
    window,
    bg = "#ffffff",
    height = 800,
    width = 1200,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

background_img = PhotoImage(file = f"background.png")
background = canvas.create_image(
    602.0, 380.0,
    image=background_img)

img0 = PhotoImage(file = f"img0.png") #Play Button
b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked,
    relief = "flat")

b0.place(
    x = 956, y = 25,
    width = 60,
    height = 60)

img1 = PhotoImage(file = f"img1.png") #Stop Button
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = exit,
    relief = "flat")

b1.place(
    x = 1042, y = 25,
    width = 60,
    height = 60)

img2 = PhotoImage(file = f"img2.png") #Browse Save Destination Button
b2 = Button(
    image = img2,
    borderwidth = 0,
    highlightthickness = 0,
    command = open_save_dir,
    relief = "flat")

b2.place(
    x = 657, y = 192,
    width = 124,
    height = 37)

img3 = PhotoImage(file = f"img3.png") #Read CSV Button
b3 = Button(
    image = img3,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked,
    relief = "flat")

b3.place(
    x = 139, y = 297,
    width = 124,
    height = 37)

img4 = PhotoImage(file = f"img4.png") #Read Firebase Button
b4 = Button(
    image = img4,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_importDatabase,
    relief = "flat")

b4.place(
    x = 139, y = 557,
    width = 124,
    height = 37)

entry0_img = PhotoImage(file = f"img_textBox0.png") #Gateway A Latitude
entry0_bg = canvas.create_image(
    1024.5, 198.5,
    image = entry0_img)

entry0 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry0.place(
    x = 951.0, y = 180,
    width = 147.0,
    height = 35)

entry1_img = PhotoImage(file = f"img_textBox1.png") #Gateway A Longitude
entry1_bg = canvas.create_image(
    1024.5, 240.5,
    image = entry1_img)

entry1 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry1.place(
    x = 951.0, y = 222,
    width = 147.0,
    height = 35)

entry2_img = PhotoImage(file = f"img_textBox2.png") #Gateway B Latitude
entry2_bg = canvas.create_image(
    1024.5, 329.5,
    image = entry2_img)

entry2 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry2.place(
    x = 951.0, y = 311,
    width = 147.0,
    height = 35)

entry3_img = PhotoImage(file = f"img_textBox3.png") #Gateway B Longitude
entry3_bg = canvas.create_image(
    1024.5, 371.5,
    image = entry3_img)

entry3 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry3.place(
    x = 951.0, y = 353,
    width = 147.0,
    height = 35)

entry4_img = PhotoImage(file = f"img_textBox4.png") #Gateway C Latitude
entry4_bg = canvas.create_image(
    1024.5, 461.5,
    image = entry4_img)

entry4 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry4.place(
    x = 951.0, y = 443,
    width = 147.0,
    height = 35)

entry5_img = PhotoImage(file = f"img_textBox5.png") #Gateway C Longitude
entry5_bg = canvas.create_image(
    1024.5, 503.5,
    image = entry5_img)

entry5 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry5.place(
    x = 951.0, y = 485,
    width = 147.0,
    height = 35)

entry6_img = PhotoImage(file = f"img_textBox6.png") #Mobile Node Latitude
entry6_bg = canvas.create_image(
    1024.5, 648.5,
    image = entry6_img)

entry6 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry6.place(
    x = 951.0, y = 630,
    width = 147.0,
    height = 35)

entry7_img = PhotoImage(file = f"img_textBox7.png") #Mobile Node Longitude
entry7_bg = canvas.create_image(
    1023.5, 690.5,
    image = entry7_img)

entry7 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry7.place(
    x = 950.0, y = 672,
    width = 147.0,
    height = 35)

entry8_img = PhotoImage(file = f"img_textBox8.png") #Save Destination Entry
entry8_bg = canvas.create_image(
    497.0, 210.5,
    image = entry8_img)

entry8 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry8.insert(END, save_destination_default)

entry8.place(
    x = 355.0, y = 192,
    width = 284.0,
    height = 35)

entry9_img = PhotoImage(file = f"img_textBox9.png") #n Constant Entry
entry9_bg = canvas.create_image(
    464.5, 308.5,
    image = entry9_img)

entry9 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry9.insert(END, n_default)

entry9.place(
    x = 397.0, y = 290,
    width = 135.0,
    height = 35)

entry10_img = PhotoImage(file = f"img_textBox10.png") #Circumference points entry
entry10_bg = canvas.create_image(
    703.5, 308.5,
    image = entry10_img)

entry10 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry10.insert(END, points_default)

entry10.place(
    x = 636.0, y = 290,
    width = 135.0,
    height = 35)

entry11_img = PhotoImage(file = f"img_textBox11.png") #COM Port Entry
entry11_bg = canvas.create_image(
    478.0, 353.5,
    image = entry11_img)

entry11 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry11.insert(END, com_default)

entry11.place(
    x = 424.0, y = 335,
    width = 108.0,
    height = 35)

entry12_img = PhotoImage(file = f"img_textBox12.png") #Baud rate entry
entry12_bg = canvas.create_image(
    478.0, 398.5,
    image = entry12_img)

entry12 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry12.insert(END, baud_default)

entry12.place(
    x = 424.0, y = 380,
    width = 108.0,
    height = 35)

entry13_img = PhotoImage(file = f"img_textBox13.png") #CSV Startrow entry
entry13_bg = canvas.create_image(
    199.0, 214.5,
    image = entry13_img)

entry13 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry13.place(
    x = 145.0, y = 196,
    width = 108.0,
    height = 35)

entry14_img = PhotoImage(file = f"img_textBox14.png") #CSV Endrow entry
entry14_bg = canvas.create_image(
    199.0, 258.5,
    image = entry14_img)

entry14 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry14.place(
    x = 145.0, y = 240,
    width = 108.0,
    height = 35)

entry15_img = PhotoImage(file = f"img_textBox15.png") #Firebase date entry
entry15_bg = canvas.create_image(
    199.0, 427.5,
    image = entry15_img)

entry15 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry15.insert(END, firebase_read_date_default)

entry15.place(
    x = 145.0, y = 409,
    width = 108.0,
    height = 35)

entry16_img = PhotoImage(file = f"img_textBox16.png") #Firebase time entry
entry16_bg = canvas.create_image(
    199.0, 472.5,
    image = entry16_img)

entry16 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry16.insert(END, firebase_read_time_default)

entry16.place(
    x = 145.0, y = 454,
    width = 108.0,
    height = 35)

entry17_img = PhotoImage(file = f"img_textBox17.png") #Firebase phone number entry
entry17_bg = canvas.create_image(
    199.0, 518.5,
    image = entry17_img)

entry17 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry17.insert(END, firebase_read_phone_default)

entry17.place(
    x = 145.0, y = 500,
    width = 108.0,
    height = 35)

window.resizable(False, False)
window.mainloop()
