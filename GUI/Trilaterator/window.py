from tkinter import *


def btn_clicked():
    print("Button Clicked")


window = Tk()

window.geometry("900x800")
window.configure(bg = "#ffffff")
canvas = Canvas(
    window,
    bg = "#ffffff",
    height = 800,
    width = 900,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

background_img = PhotoImage(file = f"background.png")
background = canvas.create_image(
    450.0, 400.0,
    image=background_img)

img0 = PhotoImage(file = f"img0.png")
b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked,
    relief = "flat")

b0.place(
    x = 601, y = 670,
    width = 212,
    height = 63)

img1 = PhotoImage(file = f"img1.png")
b1 = Button(
    image = img1,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked,
    relief = "flat")

b1.place(
    x = 378, y = 472,
    width = 124,
    height = 37)

entry0_img = PhotoImage(file = f"img_textBox0.png")
entry0_bg = canvas.create_image(
    741.5, 588.5,
    image = entry0_img)

entry0 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry0.place(
    x = 668.0, y = 570,
    width = 147.0,
    height = 35)

entry1_img = PhotoImage(file = f"img_textBox1.png")
entry1_bg = canvas.create_image(
    742.5, 546.5,
    image = entry1_img)

entry1 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry1.place(
    x = 669.0, y = 528,
    width = 147.0,
    height = 35)

entry2_img = PhotoImage(file = f"img_textBox2.png")
entry2_bg = canvas.create_image(
    742.5, 401.5,
    image = entry2_img)

entry2 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry2.place(
    x = 669.0, y = 383,
    width = 147.0,
    height = 35)

entry3_img = PhotoImage(file = f"img_textBox3.png")
entry3_bg = canvas.create_image(
    742.5, 359.5,
    image = entry3_img)

entry3 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry3.place(
    x = 669.0, y = 341,
    width = 147.0,
    height = 35)

entry4_img = PhotoImage(file = f"img_textBox4.png")
entry4_bg = canvas.create_image(
    742.5, 269.5,
    image = entry4_img)

entry4 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry4.place(
    x = 669.0, y = 251,
    width = 147.0,
    height = 35)

entry5_img = PhotoImage(file = f"img_textBox5.png")
entry5_bg = canvas.create_image(
    742.5, 227.5,
    image = entry5_img)

entry5 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry5.place(
    x = 669.0, y = 209,
    width = 147.0,
    height = 35)

entry6_img = PhotoImage(file = f"img_textBox6.png")
entry6_bg = canvas.create_image(
    742.5, 138.5,
    image = entry6_img)

entry6 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry6.place(
    x = 669.0, y = 120,
    width = 147.0,
    height = 35)

entry7_img = PhotoImage(file = f"img_textBox7.png")
entry7_bg = canvas.create_image(
    742.5, 96.5,
    image = entry7_img)

entry7 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry7.place(
    x = 669.0, y = 78,
    width = 147.0,
    height = 35)

entry8_img = PhotoImage(file = f"img_textBox8.png")
entry8_bg = canvas.create_image(
    424.5, 588.5,
    image = entry8_img)

entry8 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry8.place(
    x = 357.0, y = 570,
    width = 135.0,
    height = 35)

entry9_img = PhotoImage(file = f"img_textBox9.png")
entry9_bg = canvas.create_image(
    185.5, 588.5,
    image = entry9_img)

entry9 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry9.place(
    x = 118.0, y = 570,
    width = 135.0,
    height = 35)

entry10_img = PhotoImage(file = f"img_textBox10.png")
entry10_bg = canvas.create_image(
    218.0, 490.5,
    image = entry10_img)

entry10 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry10.place(
    x = 76.0, y = 472,
    width = 284.0,
    height = 35)

entry11_img = PhotoImage(file = f"img_textBox11.png")
entry11_bg = canvas.create_image(
    244.5, 333.5,
    image = entry11_img)

entry11 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry11.place(
    x = 177.0, y = 315,
    width = 135.0,
    height = 35)

entry12_img = PhotoImage(file = f"img_textBox12.png")
entry12_bg = canvas.create_image(
    244.5, 287.5,
    image = entry12_img)

entry12 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry12.place(
    x = 177.0, y = 269,
    width = 135.0,
    height = 35)

entry13_img = PhotoImage(file = f"img_textBox13.png")
entry13_bg = canvas.create_image(
    244.5, 242.5,
    image = entry13_img)

entry13 = Entry(
    bd = 0,
    bg = "#ffffff",
    highlightthickness = 0)

entry13.place(
    x = 177.0, y = 224,
    width = 135.0,
    height = 35)

canvas.create_text(
    44.5, 776.0,
    text = "Ready",
    fill = "#ffffff",
    font = ("Roboto-Bold", int(24.0)))

window.resizable(False, False)
window.mainloop()
