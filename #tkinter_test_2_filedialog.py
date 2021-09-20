#tkinter_test_2
from os import kill
from tkinter import * #from tkinter import everything
from tkinter import messagebox
from tkinter import filedialog #for file dialog
from PIL import ImageTk,Image #for images

#define main window
root = Tk()

#functions
def openimg(): #Do not mind this function
    global my_image
    imgwindow = Toplevel() #generate a new window
    imgwindow.pic = filedialog.askopenfilename(title="Select file", filetypes=(("jpg","*.jpg"),("All files","*.*"))) # * means any
    my_label = Label(imgwindow, text=imgwindow.pic).pack() #state the directory and filename
    my_image = ImageTk.PhotoImage(Image.open(imgwindow.pic)) #grab the image
    my_image_label = Label(imgwindow, image=my_image).pack() #show the image

def open_save_dir(): #Function for the Save Destination Button
    root.save_dest = filedialog.askdirectory(title="Choose save destination")
    save_destination.set(root.save_dest)
    global str_save_destination
    str_save_destination = save_destination.get()
    print(str_save_destination)
    return

def open_chrome_driver(): #Function for the Chrome Driver Destination Button
    root.chrome_driver_dir = filedialog.askopenfilename(title="Select Chrome Driver directory")
    chrome_driver_location.set(root.chrome_driver_dir)
    global str_chrome_driver_location 
    str_chrome_driver_location = chrome_driver_location.get()
    print(str_chrome_driver_location)
    return
    
#my_btn = Button(root, text="Open File", command=openimg).pack()

#Variables
save_destination = StringVar()
chrome_driver_location = StringVar()

#Save destination
lbl_save_dest = Label(root, textvariable=save_destination)
btn_choose_save_dest = Button(root, text="Set save destination", command=open_save_dir, padx=50,  fg="white", bg="#50C1C4")

#Chrome driver location
lbl_chromedriver_location = Label(root, textvariable=chrome_driver_location)
btn_choose_chrome_driver = Button(root, text="Set Chrome Driver location", command=open_chrome_driver, padx=50,  fg="white", bg="#50C1C4")

#Kill button
btn_kill = Button(root, text="Kill", command=root.destroy)

#Positioning
lbl_save_dest.grid(row=0,column=1)
btn_choose_save_dest.grid(row=0, column=0)
lbl_chromedriver_location.grid(row=1,column=1)
btn_choose_chrome_driver.grid(row=1,column=0)
btn_kill.grid(row=2, column=0)

root.mainloop()

#This section will only work after root.mainloop() is destoyed
str_save_destination = save_destination.get()
str_chrome_driver_location = chrome_driver_location.get()
print(str_save_destination + "\n" + str_chrome_driver_location)

#we can also open .html files with this.