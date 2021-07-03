#Reference
#https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points

#Online calculator for comparison. The code is very accurate.
#https://latlongdata.com/distance-calculator/

#########

##Start of code##
from math import radians, cos, sin, asin, sqrt

##Define Haversine functions
def haversine(lat1, lon1, lat2, lon2):

    miles = 3959.87433
    km = 6372.8

    R = km

    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))

    distance = R * c

    return distance
##End of Haversine function

###########

# Code for input
lat1 = 32.0004311
lon1 = -103.548851
lat2 = 33.374939
lon2 = -103.6041946

print(haversine(lat1, lon1, lat2, lon2))