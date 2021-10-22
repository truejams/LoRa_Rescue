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

Revised build v0.4.1 [Ianny]
- Added function for GPS to Cartesian Coordinate Converter
- Added optional function for Cartesian to GPS Converter
- Code now converts GPS coordinates of GNodes and Actual Position to Cartesian coordinates
- Removed function for Average Filter and associated lines of code
- Removed Average Filter from CSV output
  - Removed Mean Distances w/ Average Filter in Basic.csv
  - Removed Distances w/ Average Filter in Distances.csv
- Added Tolerance Filter in CSV output
  - Added Mean Coordinates w/ Tolerance Filter in Basic.csv
  - Added Coordinates w/ Tolerance Filter in Coordinates.csv
  - Added K-Means Centroids vs. Mean Coordinates w/ Tolerance Filter in K-Means.csv
  - Added K-Means Centroids vs. Coordinates w/ Tolerance Filter in K-Means.csv
- Minor formatting changes in CSV ouptuts for better visualization

Revised build v0.4.2 [Greg]
- Added error function with revised input and output

Revised build v0.4.3 [Benj]
- Removed phoneA[0:len(phoneA)]
- Disabled frequency distribution (Error is column length does not match)

Revised build v0.4.4 [Ianny]
- Changed axes names of distance behavior graph
- Added actual distance in distance behavior graph
- Changed legend location in distance behavior graph for better visualization
- Created a new function for comp_distanceAf, comp_distanceBf, and comp_distanceCf called actualDist

Revised build v0.4.5 [Ianny]
- Enabled frequency distribution
- Fixed syntax error in frequency distribution graph code

Revised build v0.4.6 [Ianny]
- In the cartToGPS() function, the types of output lat and lon were changed to lists
- Mapped the K-means graph in an html file using the folium library

Revised build v0.4.7 [Ianny]
- Removed sklearn, serial, and datetime libraries
- Removed listenforData() function and associated variables
- Changed importCSV() function to Benj's Read_rawData_test.py with additional outputs
    - The code procures data only from rawData.csv
- "Converts" Folium Map (.html) to image file (.png)
    - Requires chromedriver.exe which opens the HTML map in google chrome
    - Then, Selenium snapshots the opened webpage
    - Change browser_driver directory accordingly 
    - Download chromedriver at: https://sites.google.com/a/chromium.org/chromedriver/downloads

Revised build v0.4.8 [Ianny]
- The browser_driver is the folder directory where chromedriver.exe is contained
- Converted all output images to .jpg for minimal file size
- Brought back the serial and datetime libraries, and the listenforData() function
- Changed date and time naming convention from ';' to '-'
- Adjusted title locations of RawTrilateration, FiltTrilateration, and K-Means plot
- Added missing round() line for distSeriesA['Distance'] (used in Frequency Distribution plot)
- Added a function for saving the output image files to Firebase Storage
    - Added and used a Firebase Web App called 'LoRa Rescue Storage' for this
- Used the os library to change the working directory for Pyrebase

Revised build v0.4.9 [Ianny]
- The browser_driver now points to chromedriver.exe directly
- If ever, manually download the latest stable version of chromedriver 
    - New Download Link: https://sites.google.com/chromium.org/driver/
    - Old link will be deprecated soon

Revised build v0.5.0
Required libraries to install:
 - pip install tk
 - pip install sseclient
Changelog:
 - Added firebase database support

# LoRa_Gateway_Calibration Build Logs
Use information
- To use the code no editing of the arduino code will be necessary
- Simply plug the gateway and transmit data from the mobile node
- It is noted that only gateway A shall be used for calibration.

Build v0.1
- This code has been tested indoors with LOS environment
- The mobile node was placed 2.5 meters away
- The result showed an optimal path loss exponent of 9.9

Build v0.2
- Incorporated haversine formula for calibration
