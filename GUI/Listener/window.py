import tkinter as tk
from tkinter import *
from tkinter import ttk

import os

os.chdir(os.path.dirname(os.path.abspath(__file__))) #Set .py location as active directory (NEEDED for GUI)

def btn_clicked():
    print("Button Clicked")
    statusLabel.config(text='Listening...') #This is how we change the status text in the bottom left.
    com = entry0.get()
    print(com)

window = Tk()

window.geometry("400x350")
window.configure(bg = "#ffffff")
canvas = Canvas(
    window,
    bg = "#ffffff",
    height = 350,
    width = 400,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

background_img = PhotoImage(file = f"background.png") 
background = canvas.create_image(
    200.0, 175.0,
    image=background_img)

img0 = PhotoImage(file = f"img0.png") #Start button
b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked,
    relief = "flat")

b0.place(
    x = 173, y = 48,
    width = 60,
    height = 60)

img1 = PhotoImage(file = f"img1.png") #Stop button
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked,
    relief = "flat")

b1.place(
    x = 259, y = 48,
    width = 60,
    height = 60)

"Uncomment the section below to generate the rounded corner image for the dropdown"

# entry0_img = PhotoImage(file = f"img_textBox0.png") 
# entry0_bg = canvas.create_image(
#     255.0, 200.5,
#     image = entry0_img)


entry0 = ttk.Combobox(values=["jamol","babol"]) #Dropdown for COM port

entry0.place(
    x = 201.0, y = 182,
    width = 108.0,
    height = 35)

statusLabel = ttk.Label()

statusLabel.place(
    x=12.0, y=320.0
)

statusLabel.config(text='Ready', foreground='#ffffff', background='#4BC1C3', font=("Roboto-Bold", int(12.0)))

window.resizable(False, False)
window.mainloop()
