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

def btn_clicked():
    print("Button Clicked")

def open_save_dir(): #Function for the Save Destination Button
    global save_path
    save_path = filedialog.askdirectory(title="Choose save destination")
    entry8.delete(0, END)
    entry8.insert(0, save_path)
    
    save_path = save_path + '/'
    print("Save destination set to: " + save_path)
    return

def btn_readcsv(): #Function to read csv using importCSV function.
    global save_path
    startrow = int(entry13.get())
    endrow = int(entry14.get())
    rssiA, rssiB, rssiC, dtn, phone = importCSV(save_path, startrow, endrow)
    print(dtn)

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

# Firebase Web App Configuration
LoraRescueStorage = {'apiKey': "AIzaSyAN2jdAfGBhbPz446Lho_Jmu2eysU6Hvqw",
    'authDomain': "lora-rescue.firebaseapp.com",
    'databaseURL': "https://lora-rescue-default-rtdb.asia-southeast1.firebasedatabase.app",
    'projectId': "lora-rescue",
    'storageBucket': "lora-rescue.appspot.com",
    'messagingSenderId': "295122276311",
    'appId': "1:295122276311:web:68ce5d4d4cd6763103c592",
    'measurementId': "G-MCPTP8HPLK"}

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

img0 = PhotoImage(file = f"img0.png")
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

img1 = PhotoImage(file = f"img1.png")
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked,
    relief = "flat")

b1.place(
    x = 1042, y = 25,
    width = 60,
    height = 60)

entry0_img = PhotoImage(file = f"img_textBox0.png")
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

entry1_img = PhotoImage(file = f"img_textBox1.png")
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

entry2_img = PhotoImage(file = f"img_textBox2.png")
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

entry3_img = PhotoImage(file = f"img_textBox3.png")
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

entry4_img = PhotoImage(file = f"img_textBox4.png")
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

entry5_img = PhotoImage(file = f"img_textBox5.png")
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

entry6_img = PhotoImage(file = f"img_textBox6.png")
entry6_bg = canvas.create_image(
    1024.5, 659.5,
    image = entry6_img)

entry6 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry6.place(
    x = 951.0, y = 641,
    width = 147.0,
    height = 35)

entry7_img = PhotoImage(file = f"img_textBox7.png")
entry7_bg = canvas.create_image(
    1023.5, 701.5,
    image = entry7_img)

entry7 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry7.place(
    x = 950.0, y = 683,
    width = 147.0,
    height = 35)

entry8_img = PhotoImage(file = f"img_textBox8.png")
entry8_bg = canvas.create_image(
    504.0, 210.5,
    image = entry8_img)

entry8 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry8.place(
    x = 362.0, y = 192,
    width = 284.0,
    height = 35)

img2 = PhotoImage(file = f"img2.png")
b2 = Button(
    image = img2,
    borderwidth = 0,
    highlightthickness = 0,
    command = open_save_dir,
    relief = "flat")

b2.place(
    x = 664, y = 192,
    width = 124,
    height = 37)

img3 = PhotoImage(file = f"img3.png")
b3 = Button(
    image = img3,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_readcsv,
    relief = "flat")

b3.place(
    x = 139, y = 254,
    width = 124,
    height = 37)

entry9_img = PhotoImage(file = f"img_textBox9.png")
entry9_bg = canvas.create_image(
    471.5, 390.5,
    image = entry9_img)

entry9 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry9.place(
    x = 404.0, y = 372,
    width = 135.0,
    height = 35)

entry10_img = PhotoImage(file = f"img_textBox10.png")
entry10_bg = canvas.create_image(
    710.5, 390.5,
    image = entry10_img)

entry10 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry10.place(
    x = 643.0, y = 372,
    width = 135.0,
    height = 35)

entry11_img = PhotoImage(file = f"img_textBox11.png")
entry11_bg = canvas.create_image(
    485.0, 480.5,
    image = entry11_img)

entry11 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry11.place(
    x = 431.0, y = 462,
    width = 108.0,
    height = 35)

entry12_img = PhotoImage(file = f"img_textBox12.png")
entry12_bg = canvas.create_image(
    485.0, 435.5,
    image = entry12_img)

entry12 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry12.place(
    x = 431.0, y = 417,
    width = 108.0,
    height = 35)

entry13_img = PhotoImage(file = f"img_textBox13.png")
entry13_bg = canvas.create_image(
    199.0, 171.5,
    image = entry13_img)

#entry13 import CSV start row
entry13 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry13.place(
    x = 145.0, y = 153,
    width = 108.0,
    height = 35)

entry14_img = PhotoImage(file = f"img_textBox14.png")
entry14_bg = canvas.create_image(
    199.0, 215.5,
    image = entry14_img)

#entry14 import CSV end row
entry14 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry14.place(
    x = 145.0, y = 197,
    width = 108.0,
    height = 35)

window.resizable(False, False)
window.mainloop()
