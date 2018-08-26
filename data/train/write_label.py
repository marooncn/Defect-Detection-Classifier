#!/usr/bin/python
# -*- coding: UTF-8 -*- 

f = open("label.csv", "w")
for i in range(160):
    f.write('./normal/' + str(i+1)+'.jpg'+',1\n')
for i in range(160):
    f.write('./defect/' + str(i+1)+'.jpg'+',0\n')
f.close()
