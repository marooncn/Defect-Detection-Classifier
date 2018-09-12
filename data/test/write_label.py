#!/usr/bin/python
# -*- coding: UTF-8 -*- 

f = open("label.csv", "w")
for i in range(40):
    f.write('./data/test/normal/' + str(i+1)+'.jpg'+',1\n')
for i in range(40):
    f.write('./data/test/defect/' + str(i+1)+'.jpg'+',0\n')
f.close()
