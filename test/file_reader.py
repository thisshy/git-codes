# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 17:24:46 2021

@author: 11870
"""
with open('pi_digits.txt') as file_object: 
 lines = file_object.readlines()
 a=''
 for line in lines:
     a += line.rstrip()

b=input('you shnegri: ')
if b in a:
    print('yes')
