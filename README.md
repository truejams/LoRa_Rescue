# LoRa_Rescue Build Logs:

Revised build v0.3
Changelog:
Edited the listenForData function to not convert directly to distance
- rewrote to fit all data in a single csv file
- Saved the rssi instead
- Added compute for time interval of data
- Added an error.csv where all the bad data are stored
Added a function to convert rssi to distance
Added a function for the tolerance filter
Changed spacing on the last part

Revised build v0.3.1
Changelog:
Added Haversine function made be Benj
- Created new variables to stare longitude and latitude values
- Have not implemented the function for error checking
