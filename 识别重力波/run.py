from tkinter import *
from tkinter import ttk

import tkinter.messagebox as messagebox
from tkinter.filedialog import askdirectory

from testnewd import runClassFier

class MyFrame(Frame):
    def __init__(self,master=None):
        Frame.__init__(self,master)
        self.grid()
        self.init()
        self.createWidgets()
    
    #初始化用到的变量
    def init(self):
        self.opname = ""
        self.opath = StringVar()
        
        self.ipname = ""
        self.ipath = StringVar()

        self.pbarValue = IntVar()

        self.ptvar = StringVar()

    #展示视图
    def createWidgets(self):
        self.fileIpath = Entry(self,textvariable=self.ipath)
        self.fileIpath.grid(row=0,column=1)
        self.selectIbutton = Button(self,text="输入文件夹",command=self.selectIpath)
        self.selectIbutton.grid(row=0,column=2)

        self.fileOpath = Entry(self,textvariable=self.opath)
        self.fileOpath.grid(row=1,column=1)
        self.selectObutton = Button(self,text="输出文件夹",command=self.selectOpath)
        self.selectObutton.grid(row=1,column=2)

        self.pbar = ttk.Progressbar(self,length=200,mode="determinate")
        self.pbar.grid(row=2,column=1)
        self.pbar["value"] = 0

        self.pbar["variable"] = self.pbarValue
        
        self.ptext = Label(self,textvariable=self.ptvar,font=(10))
        self.ptext.grid(row=3,column=1)

        self.starButtoon = Button(self,text="开始",command = self.starClassfier)
        self.starButtoon.grid(row=4,column=1)

    
    #选择输入路径
    def selectIpath(self):
        self.ipname = askdirectory()
        print(self.ipname)
        self.ipath.set(self.ipname)
    
    #选择输出路径
    def selectOpath(self):
        self.opname = askdirectory()
        self.opath.set(self.opname)

    def starClassfier(self):
        for st,total in runClassFier(self.ipname,self.opname):
            count = int((st/total)*100)
            print(st,total,count)
            self.ptvar.set(str(st)+"/"+str(total))
            self.pbarValue.set(count)
            self.update()


app = MyFrame()
app.master.title("图片分类")
app.mainloop()