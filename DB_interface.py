from tkinter import *
from tkinter import filedialog
import tkinter
import io
import csv
import codecs

def OpenFile(text):
    textbox=text
    filename = filedialog.askopenfile()
    textbox.delete('1.0', END)
    with open(filename.name, encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            temp=" ".join(row)
            textbox.insert(END,temp)
            textbox.insert(END,"\n")
            
def create_window():
    window = Toplevel()
    canvas = Canvas(window, width = 1100, height = 1200)
    canvas.pack(expand = YES, fill = BOTH)
    gif1 = PhotoImage(file = 'test.gif')
    canvas.create_image(50, 10, image = gif1, anchor = NW)
    canvas.gif1=gif1
    
    
def execute(text):
    textbox=text
    textbox.delete('1.0', END)
    textbox.insert(END,"File Executed!!")  

def clear(text):
    textbox=text
    textbox.delete('1.0', END)  

root = Tk()
root.geometry("1200x750+200+100")
root.resizable(0,0)
root.title("Twitter Machine Learning Interface")
Label(text="Twitter Data", fg="blue").grid(row=0,column=0,columnspan=2)
textbox1= Text(root, width=70,height=35)
textbox1.grid(row=1,rowspan=9, column=0,columnspan=2,padx=20)
Button(text="Click Here to Upload Twitter Data",fg="green", command= lambda:OpenFile(textbox1)).grid(row=10, column=0,columnspan=2,pady=10)
textbox1.insert(END,"this is where the data goes")
Button(text="Execute", bg="#666699", fg="white", command= lambda:execute(textbox1)).grid(row=12, column=0,columnspan=2,pady=10)
Button(text="RESET",bg="red", fg="white", command=lambda:clear(textbox1)).grid(row=14, column=0,columnspan=2)
Label(text="Machine Analysis",fg="blue").grid(row=0,column=2,columnspan=2)
textbox2= Text(width=70,height=35)
textbox2.grid(row=1,rowspan=9, column=2,columnspan=2,padx=20)
textbox2.insert(END,"this is where AI data goes")
Button(text="Click Here To Upload File To Machine Learning",fg="green", command=lambda:OpenFile(textbox2)).grid(row=10,column=2, columnspan=2,pady=10)
Button(text="Execute",bg="#666699", fg="white", command=create_window).grid(row=12, column=2,columnspan=2,pady=10)
Button(text="RESET",bg="red", fg="white",  command=lambda:clear(textbox2)).grid(row=14, column=2,columnspan=2)

root.mainloop()  


