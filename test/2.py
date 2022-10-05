# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:16:25 2021

@author: 11870
"""

class gw():
    def __init__(self,gw_meg,last_time,ele_thing):
       
        self.gw_meg=gw_meg
        self.last_time=last_time
        self.ele_thing=ele_thing
    def get_descriptive_name(self):
        for a in self.gw_meg:
            print(a)
        print(self.last_time)
        if self.ele_thing==True:
            print('有电磁体')
        else:
            print('无电磁体')
    def importance (self):
        if self.ele_thing==True:
            d=0.1*self.gw_meg[0]+0.1*self.gw_meg[1]+0.1*self.last_time+5
        else :
            d=0.1*self.gw_meg[0]+0.1*self.gw_meg[1]+0.1*self.last_time-5
        return d


gw_1=gw([1,2],3,False)
gw_1.get_descriptive_name()
print(gw_1.importance())

        
        