from tkinter import *
from tkinter import ttk

import pandas as pd

# save_destination = "C:\\Users\\Benj\\Desktop\\LoRa_Rescue\\10-30-21_Data\\"
save_destination = "C:\\LoRa_Rescue\\10-30-21_Data\\"
# Read RawData.csv Configuration
# In excel, the first row is treated as Row 0
    # Basically, subtract 1 from excel row number
################## CHANGE THIS ACCORDINGLY ##################  

startrow = 994
endrow = 1044

def btn_clicked():
    print("Button clicked")

def btn_readcsv():
    rssiA, rssiB, rssiC, dtn, phone = importCSV(save_destination, startrow, endrow)
    print(rssiA)
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
    command = btn_clicked,
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
    x = 18, y = 215,
    width = 124,
    height = 37)

img4 = PhotoImage(file = f"img4.png")
b4 = Button(
    image = img4,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked,
    relief = "flat")

b4.place(
    x = 18, y = 169,
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

# entry12_img = PhotoImage(file = f"img_textBox12.png")
# entry12_bg = canvas.create_image(
#     485.0, 435.5,
#     image = entry12_img)

#COM PORT

entry12 = ttk.Combobox()

# entry12 = Entry(
#     bd = 0,
#     bg = "#ffffff",
#     highlightthickness = 0)

# entry12.insert(END, "babol")

entry12.place(
    x = 431.0, y = 417,
    width = 108.0,
    height = 35)

window.resizable(False, False)
window.mainloop()
