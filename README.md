# LoRa_Rescue Build Logs:

Revised build v0.3
- Edited the listenForData function to not convert directly to distance
  - rewrote to fit all data in a single csv file
  - Saved the rssi instead
  - Added compute for time interval of data
  - Added an error.csv where all the bad data are stored
- Added a function to convert rssi to distance
- Added a function for the tolerance filter
- Changed spacing on the last part

Revised build v0.3.1
- Added Haversine function made be Benj
  - Created new variables to stare longitude and latitude values
  - Have not implemented the function for error checking

# LoRa_Gateway_Calibration Build Logs
Use information
- To use the code no editing of the arduino code will be necessary
- Simply plug the gateway and transmit data from the mobile node
- It is noted that only gateway A shall be used for calibration.

Build v0.1
- This code has been tested indoors with LOS environment
- The mobile node was placed 2.5 meters away
- The result showed an optimal path loss exponent of 9.9


