from tkinter import *
from tkinter import ttk, colorchooser
from PIL import ImageGrab, ImageOps
from predictor import Predictor
import json_utils
import torch
from torchvision import transforms

class Main:
    def __init__(self, root):
        self.root = root
        self.predictor = Predictor()
        self.x = None
        self.y = None
        self.penwidth = 30
        self.drawWidgets()
        self.canv.bind('<B1-Motion>',self.paint)#drwaing the line 
        self.canv.bind('<ButtonRelease-1>',self.reset)
        
    def paint(self, e):
        if self.x and self.y:
            self.canv.create_line(self.x, self.y, e.x, e.y, width=self.penwidth, fill='black', capstyle=ROUND, smooth=True)        
        self.x = e.x
        self.y = e.y
    
    def reset(self, e):
        self.x = None
        self.y = None      

    def changeW(self, e): #to change pen width by using the slider
        self.penwidth = e
           

    def clear(self):
        self.canv.delete(ALL)
        
    #def change_model(x):
        #self.predictor.change model???

    def canvas_prediction(self):    
        #get canvas' image
        x=self.root.winfo_rootx()+self.canv.winfo_x()
        y=self.root.winfo_rooty()+self.canv.winfo_y()
        x1=x+self.canv.winfo_width()
        y1=y+self.canv.winfo_height()
        canvas_image = ImageGrab.grab().crop((x,y,x1,y1))
        canvas_image = ImageOps.grayscale(canvas_image)
        canvas_image = ImageOps.invert(canvas_image)

        #downscale to mnist 28x28
        canvas_image = canvas_image.resize((28, 28), resample=4, box=None)
        to_tensor = transforms.ToTensor()
        x = to_tensor(canvas_image)
        #add batch dimension
        x = torch.unsqueeze(x, dim=0)
        
        #get currently selected model's id from combobox, use it to get model's key and use it to load the model
        self.predictor.load_model(list(self.predictor.models_dict)[self.model_combobox.current()])
        
        results = self.predictor.predict(x) * 100

        self.updateTable(results)
    
        
    def updateTable(self, results):
        for i in range(10):                 
            self.results = Entry(self.prediction, width=9,
                           font=('Arial',12,'bold'))
             
            self.results.grid(row=i, column=1)
            self.results.insert(END, str(round(results[0][i].item(), 3)) + " %")
    
    def drawWidgets(self):
        #create frame for table and buttons frame
        self.display = Frame(self.root,padx = 10,pady = 10)
        self.display.pack(side=RIGHT)
        
        #create table to display prediction values
        self.prediction = Frame(self.display, padx = 10, pady = 10)
        Label(self.display, text='Prediction:', font=('arial 12')).pack()
        for i in range(10):                 
            self.results = Entry(self.prediction, width=4,
                           font=('Arial',12,'bold'))
            self.results.grid(row=i, column=0)
            self.results.insert(END, str(i)+": ")
            
            self.results = Entry(self.prediction, width=9,
                           font=('Arial',12,'bold'))
            self.results.grid(row=i, column=1)
            self.results.insert(END, "")

        self.prediction.pack(side=TOP)
        
        
        #models combobox
        Label(self.display, text='Selected model:', font=('arial 12')).pack()
        self.model_combobox = ttk.Combobox(self.display)
        #models_dict keys are displayed as selectable options
        self.model_combobox['values'] = list(self.predictor.models_dict)
        # prevent typing a value
        self.model_combobox['state'] = 'readonly'
        #set default value to models_dict's 1st entry
        self.model_combobox.set(list(self.predictor.models_dict)[0])
        self.model_combobox.pack(padx=5, pady=5)
        
        
        #create frame with buttons
        self.buttons = Frame(self.display, padx = 10, pady = 10)
        self.buttons.pack(side=BOTTOM)
        self.predict_button = Button(self.buttons, text ="Predict", command = self.canvas_prediction, padx = 5, pady = 5)
        self.predict_button.pack(side=LEFT)
        self.clear_button = Button(self.buttons, text ="Clear", command = self.clear, padx = 5, pady = 5)
        self.clear_button.pack(side=LEFT)
        
        #create pen_width slider
        self.controls = Frame(self.display, padx = 5, pady = 5)
        Label(self.controls, text='Pen Width:', font=('arial 12')).grid(row=0,column=0)
        self.slider = ttk.Scale(self.controls,from_= 20, to = 50,command=self.changeW,orient=VERTICAL)
        self.slider.set(self.penwidth)
        self.slider.grid(row=0,column=1,ipadx=30)
        self.controls.pack()
                
        #create canvas
        self.canv = Canvas(self.root,width=28*16,height=28*16,bg='white')
        self.canv.pack(fill=BOTH,expand=True)

        #add option menu #TODO add more options
        menu = Menu(self.root)
        self.root.config(menu=menu)
        menu.add_command(label='Add model',command=self.popup) 

        
    def popup(self):
        self.w=PopupWindow(self.root, self.callback) 
        self.root.wait_window(self.w.top)

    def callback(self, a, b, c):
        json_utils.write_to_json("settings.json", a, b, c)

#popup window with form for adding new models
class PopupWindow():
    def __init__(self, master, callback):
        top=self.top=Toplevel(master)
        top.minsize(300, 200)
        self.master = master
        self.callback = callback
        self.grid_frame = Frame(top, padx = 10, pady = 10)
        self.grid_frame.pack(side=TOP)
        # configure the grid
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=3)
        
        
        # username
        model_name_label = ttk.Label(self.grid_frame, text="Model name:")
        model_name_label.grid(column=0, row=0, sticky=W, padx=5, pady=5)
        
        self.model_name = ttk.Entry(self.grid_frame)
        self.model_name.grid(column=1, row=0, sticky=E, padx=5, pady=5)
        
        # password
        model_file_label = ttk.Label(self.grid_frame, text="Model's file name:")
        model_file_label.grid(column=0, row=1, sticky=W, padx=5, pady=5)
        
        self.model_file = ttk.Entry(self.grid_frame)
        self.model_file.grid(column=1, row=1, sticky=E, padx=5, pady=5)
        
        # password
        weights_file_label = ttk.Label(self.grid_frame, text="Model's weights file name:")
        weights_file_label.grid(column=0, row=2, sticky=W, padx=5, pady=5)
        
        self.weights_file = ttk.Entry(self.grid_frame)
        self.weights_file.grid(column=1, row=2, sticky=E, padx=5, pady=5)
        
        # ok button
        ok_button = ttk.Button(top, text="OK", command=self.submit)
        ok_button.pack()
        # back button
        back_button = ttk.Button(top, text="Back", command=self.cancel)
        back_button.pack()
        
        #freeze main window until this one is closed
        top.grab_set()
        
    def submit(self):
        self.callback(self.model_name.get(), self.model_file.get(), self.weights_file.get())
        self.top.destroy()
        
    def cancel(self):
        self.top.destroy()

if __name__ == '__main__':
    root = Tk()
    Main(root)
    root.title('Application')
    root.mainloop()
