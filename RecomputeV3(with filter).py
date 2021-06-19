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

distanceAf = list()
distanceBf = list()
distanceCf = list()

###### CHANGE THIS FOR YOUR DIRECTORY
################################################################
save_destination = "C:\\Users\\Grego\\LoRa Rescue Data 2\\"

with open(save_destination+'gatewayAt.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            phoneA = row[1]
        elif line_count > 0 and line_count < 59:
            distanceAf.append(row[1])
        elif line_count == 61:
            dtn = row[1]
            dtn = dtn.replace(':',';')
        line_count += 1

with open(save_destination+'gatewayBt.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count > 0 and line_count < 59:
            distanceBf.append(row[1])
        line_count += 1

with open(save_destination+'gatewayCt.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count > 0 and line_count < 59:
            distanceCf.append(row[1])
        line_count += 1

for i in range(len(distanceAf)):
    distanceAf[i] = float(distanceAf[i])
    distanceBf[i] = float(distanceBf[i])
    distanceCf[i] = float(distanceCf[i])

#samples = 62 * 6 - 1 #how many samples to collect per gateway * number of loops total
#line = 0 #start at 0 because our header is 0 (not real data)

if 1 == 1:
    # Run the trilateration algorithm here
    # Pwede mo i-add dito yung trilat code.
    # Use distanceAf, distanceBf, distanceCf.
    # print("The program is now trilaterating...\n")
    # YaniCode Starts Here
    ###############TRILATERATION#############
    # Convert Distances from each GNode to numpy arrays
    distanceAf = np.array(distanceAf)
    distanceBf = np.array(distanceBf)
    distanceCf = np.array(distanceCf)

    # GNode Coordinates
    # Format: A B C
    xg = np.array([0,9.0,-34.0])
    yg = np.array([0,196.0,-79.0])

    # Actual Node Position
    xAct = -24        #Target x-coordinate
    yAct = 0     #Target y-coordinate

    Rab = np.sqrt((xg[1]**2)+(yg[1]**2))
    Rbc = np.sqrt(((xg[2]-xg[1])**2)+((yg[2]-yg[1])**2))

    Ran = np.sqrt((xAct**2)+(yAct**2))
    Rnc = np.sqrt(((xg[2]-xAct)**2)+((yg[2]-yAct)**2))

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

    if yg[2] == 0:
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

    # Trilateration Calculations
    def trilatEqn(z):
        x = z[0]
        y = z[1]
        w = z[2]

        F = np.empty((3))
        F[0] = ((x-xg[0])**2) + ((y-yg[0])**2) - (dA**2)
        F[1] = ((x-xg[1])**2) + ((y-yg[1])**2) - (dB**2)
        F[2] = ((x-xg[2])**2) + ((y-yg[2])**2) - (dC**2)
        return F
    zGuess = np.array([1,1,1])

    x = list()
    y = list()

    AfAve = sum(distanceAf)/len(distanceAf)
    BfAve = sum(distanceBf)/len(distanceBf)
    CfAve = sum(distanceCf)/len(distanceCf)

    ## THIS IS A FILTER ///////////////////
    # Adds way to remove outliers in the distances
    errorTolerance = 50
    i = 0
    while i < 60:
        e = 0
        if distanceAf[i] > AfAve + errorTolerance or distanceAf[i] < AfAve - errorTolerance:
            e = 1
        if distanceBf[i] > BfAve + errorTolerance or distanceBf[i] < BfAve - errorTolerance:
            e = 1
        if distanceCf[i] > CfAve + errorTolerance or distanceCf[i] < CfAve - errorTolerance:
            e = 1
        if e != 0:
            distanceAf = np.delete(distanceAf,i)
            distanceBf = np.delete(distanceBf,i)
            distanceCf = np.delete(distanceCf,i)
            if i == len(distanceAf)-1:
                i = 60
                continue
        dA = distanceAf[i]
        dB = distanceBf[i]
        dC = distanceCf[i]
        z = fsolve(trilatEqn,zGuess)
        x.append(z[0])
        y.append(z[1])
        i += 1
        if i == len(distanceAf)-1:
            i = 60

    # Compute for the average distances based on the received data
    AfAve = sum(distanceAf)/len(distanceAf)
    BfAve = sum(distanceBf)/len(distanceBf)
    CfAve = sum(distanceCf)/len(distanceCf)
    
    # Plot the data for trilateration w/o the filters
    plt.figure(1)
    plt.scatter(x, y, label='Phone Node Locations', cmap='brg', s=20)
    plt.scatter(xg, yg, marker='1', label='GNode Locations', c='black', s=20)
    plt.title(dtn + ' 0' + phoneA[0:len(phoneA)-1]  + ' Trilateration')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.legend()
    # CHANGE
    plt.savefig(save_destination + dtn 
                + ' 0' + phoneA[0:len(phoneA)-1] + ' Trilateration.jpg') #Change Directory Accordingly

    ###############K-Means Clustering#############
    #K-means Clustering won't be performed if there is only 1 set of coordinates in the Dataset.
    if len(x)<2:
        quit()

    unformattedX = x
    unformattedY = y
    xAve = np.mean(x)
    yAve = np.mean(y)

    ## THIS IS A FILTER ///////////////////
    # Adds way to remove outliers in the coordinates
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
    
    #Create numpy array containing (x,y) coordinates
    data=np.array([[x[0],y[0]]])
    for i in range(1,len(x)):
        data=np.append(data,[[x[i],y[i]]], axis=0)
    data = np.unique(data, axis=0) #Eliminate Duplicates in data
    xAve = np.mean(data[:,0])
    yAve = np.mean(data[:,1])
    inertia = [] #aka Sum of Squared Distance Errors

    #Compute for inertias for every possible number of clusters
    for i in range(1,len(data)):
        kmeans = KMeans(n_clusters=i).fit(data)
        inertia.append(kmeans.inertia_)

    #Determine optimal Number of Clusters based on Elbow
    elbow = KneeLocator(range(1,len(data)),inertia, curve='convex', direction='decreasing')

    print('Optimal Number of Clusters is', elbow.knee)
    kmeans = KMeans(n_clusters=elbow.knee, n_init=5).fit(data) #Perform K-means with elbow no. of clusters

    #Elbow Plot
    plt.figure(2)
    plt.plot(range(1,len(data)), inertia)
    plt.xlabel('No. of Clusters')
    plt.ylabel('Sum of Squared Distances')
    plt.title(dtn + ' 0' + phoneA[0:len(phoneA)-1]  + ' Elbow Graph')
    # CHANGE
    plt.savefig(save_destination + dtn 
                + ' 0' + phoneA[0:len(phoneA)-1] + ' Elbow.jpg') #Change Directory Accordingly

    #K-Means Plot
    plt.figure(3)
    plt.scatter(data[:,0],data[:,1], c=kmeans.labels_, label = 'Phone Node Locations', cmap='brg', s=5)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c=list(range(1,elbow.knee+1)), 
                marker = 'x', label = 'Cluster Centers', cmap='brg', s=30)
    plt.scatter(xg, yg, marker='1', label='GNode Locations', c='black', s=30)
    plt.scatter(xAve, yAve, marker='^', label='Average Point', c='black', s=30)
    plt.scatter(xAct, yAct, marker='*', label='Actual Point', c='green', s=30)
    plt.grid(linewidth=1, color="w")
    ax = plt.gca()
    ax.set_facecolor('gainsboro')
    ax.set_axisbelow(True)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title(dtn + ' 0' + phoneA[0:len(phoneA)-1]  + ' K-Means')
    plt.legend()
    plt.savefig(save_destination + dtn 
                + ' 0' + phoneA[0:len(phoneA)-1] + ' K-Means.jpg') #Change Directory Accordingly

    ###############Error Computations############
    #Computed Position vs. Actual Position
    compVact = list()
    for i in range(len(x)):
        compVact.append(np.sqrt((x[i]-xAct)**2+(y[i]-yAct)**2))

    #Computed distanceAf, Bf, Cf
    comp_distanceAf = list()
    comp_distanceBf = list()
    comp_distanceCf = list()

    comp_distanceAf = np.sqrt((xAct**2)+(yAct**2))
    comp_distanceBf = np.sqrt(((xAct-xg[1])**2)+((yAct-yg[1])**2))
    comp_distanceCf = np.sqrt(((xAct-xg[2])**2)+((yAct-yg[2])**2))
    

    #K-means centroid vs. Average Point (dataset average)
    centVave = np.sqrt((kmeans.cluster_centers_[:,0]-xAve)**2+(kmeans.cluster_centers_[:,1]-yAve)**2)

    #Computed Position vs. K-means centroid
    compVcent = np.sqrt([(data[:,0]-kmeans.cluster_centers_[0,0])**2+(data[:,1]-kmeans.cluster_centers_[0,1])**2])
    for i in range(1,len(kmeans.cluster_centers_)):
        distance = np.sqrt([(data[:,0]-kmeans.cluster_centers_[i,0])**2+(data[:,1]-kmeans.cluster_centers_[i,1])**2])
        compVcent = np.append(compVcent,distance,axis=0)

    ###############CSV Writing############
    with open(save_destination+'Recompute.csv', mode='a') as logs:
        logswrite = csv.writer(logs, dialect='excel', lineterminator='\n')
        logswrite.writerow(['Actual Position','',xAct,yAct])
        logswrite.writerow(['gnodeA','gnodeB','gnodeC'])
        logswrite.writerows([[np.append(xg[0],yg[0]), np.append(xg[1],yg[1]), np.append(xg[2],yg[2])]])
        logswrite.writerow(['Computed Distances from Gnodes'])
        logswrite.writerow(['A','B','C'])
        logswrite.writerow([comp_distanceAf,comp_distanceBf,comp_distanceCf])
        logswrite.writerow(['Average Points'])
        logswrite.writerow(['A','B','C','','Xave','Yave'])
        logswrite.writerow([AfAve,BfAve,CfAve,'',xAve,yAve])
        logswrite.writerow(['K-Means Centroids vs. Average Point'])
        logswrite.writerows([centVave])
        logswrite.writerow(['Optimal # of Clusters','','',elbow.knee])
        logswrite.writerow(['Time','Phone#','Ra','Rb','Rc','','Xcomp','Ycomp','I','MSE'])
        for i in range(len(unformattedX)):
            if i < len(inertia):
                logswrite.writerow([dtn,'0'+phoneA[0:len(phoneA)-1],distanceAf[i],distanceBf[i],distanceCf[i],'',x[i],y[i],inertia[i],compVact[i]])
            elif i >= len(inertia) and i <len(compVact):
                logswrite.writerow([dtn,'0'+phoneA[0:len(phoneA)-1],distanceAf[i],distanceBf[i],distanceCf[i],'',x[i],y[i],'',compVact[i]])
            else: 
                logswrite.writerow([dtn,'0'+phoneA[0:len(phoneA)-1],distanceAf[i],distanceBf[i],distanceCf[i]])

        logswrite.writerow(['Computed Position vs. K-means Centroid:'])
        rangeof_compVcent = compVcent[:,1]
        for i in range(len(rangeof_compVcent)):
            logswrite.writerows([compVcent[i,:]])
        logswrite.writerow([''])
        logswrite.writerow([''])

    #YaniCode ends here

else:
    print("Error. Time Mismatch.")
    okA = 0
    okB = 0
    okC = 0
    ok = 0
#Find a way to reload the script.