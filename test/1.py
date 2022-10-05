# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 14:57:00 2021

@author: 11870
"""

gw170817 = {'信息源': '中子星源',
            '信号持续时间': '信号持续100秒',
            '有无电磁对应体': '有电磁对应体',}
gw231211 = {'信息源': '黑洞源',
            '信号持续时间': '信号持续2秒',
            '有无电磁对应体': '无电磁对应体',}
gw251001 = {'信息源': '中子星源',
            '信号持续时间': '信号持续88秒',
            '有无电磁对应体': '有电磁对应体',}

gw_things=[gw170817,gw231211,gw251001]

def show(gw_things):
    for gw_thing in gw_things:
        print(gw_thing)
        for a,b in gw_thing.items():
            print(a+':'+b)
            
show(gw_things)

gw251011 = {'信息源': '黑洞源',
            '信号持续时间': '信号持续1秒',
            '有无电磁对应体': '无电磁对应体',}
gw_things.append(gw251011)

show(gw_things)

for gw_thing in gw_things:
    if gw_thing['信息源'] == '中子星源':
        for a,b in gw_thing.items():
            print(a+':'+b)
            
gw_things[-1]['信号持续时间']='信号持续2秒'

show(gw_things)

for gw_thing in gw_things:
    if gw_thing['信息源'] == '黑洞源':
        gw_things.remove(gw_thing)
        
show(gw_things)


            







