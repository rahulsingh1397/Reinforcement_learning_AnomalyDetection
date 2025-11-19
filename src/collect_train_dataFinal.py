# this code is to merge all the data obtained from different log files into one model and write it to another csv
# for visualization

import numpy as np
import csv
import pickle
import json
# from dataPreprocessRawDict import*
from dataAggregateRawDict import*
from datetime import datetime, timedelta
from time import process_time

model_total = {}

# if already saved json data to be used
# with open("saveDictRawTrain18.json", 'r') as f:
#     model_total = json.load(f)

destinationHosts = []

state = input("Is the training model running for the first time? [Y/N]: ")

if state == 'N':
    with open('../data/destinationlabel.csv', 'r') as destFile:
        next(destFile) # Skipping header name
        reader = csv.reader(destFile, delimiter = ",")
        for row in reader:
            destinationHosts.append(row[1])

filePath = "../data/SBM-2023-07-05/SBM-2023-07-04.csv"
init_date = datetime.strptime("2023-06-20", "%Y-%m-%d")
# fileNames = ["SBM-2023-06-20.csv", "SBM-2023-06-21.csv", "SBM-2023-06-24.csv", "SBM-2023-07-02.csv"]
# fileNames = []
num_wd = 0
num_sat = 0
num_sun = 0

tic = process_time()
# for fileName in fileNames:
for i in range(3):
    new_date = init_date + timedelta(days=i)
    
    if new_date.weekday() <=4:
        wd = 'WD'

    elif new_date.weekday() ==5:
        wd = 'Sat'

    else:
        wd = 'Sun'

    fileName = "SBM-" + (new_date).strftime("%Y-%m-%d") + ".csv"
    # fileName = fileNames[i]

    model_total, destinationHosts = writeToModel(model_total, destinationHosts, filePath)

        # with open("model_final.csv", 'w', newline = '') as csvFile, open("destinationLabel.csv", 'w', newline = '') as labelFile:
                
        #     # key_names = ["UserName", "UserLabel", "StartDate", "IntervalCounter", "SourceAddress", "SourceCounter",\
        #     #                     "DestinationHost", "DestinationCounter"]
        #     key_names = ["UserName", "UserLabel", "IntervalCounter", "SourceAddress", "SourceCounter",\
        #                         "DestinationHost", "DestinationCounter"]
        #     writedata = csv.DictWriter(csvFile, fieldnames = key_names)
        #     writedata.writeheader()  # writing the header

        #     labeldata = csv.writer(labelFile, delimiter = ",")
        #     labeldata.writerow(["Label", "Host Name"])

    for key in model_total.keys():
        model_total[key][wd]['DayCounter'] += 1

# writing destination label data
with open("../data/destinationLabel_new.csv", 'w', newline = '') as labelFile:
    labeldata = csv.writer(labelFile, delimiter = ",")
    labeldata.writerow(["Label", "Host Name"])
    
    for i in range(len(destinationHosts)):
        labeldata.writerow([i, destinationHosts[i]])

# Updating the average values for intervals, sources and destinations

for key in model_total.keys():

    num_wd = len(model_total[key]['WD']['IntervalCounter']) - 1 # length of dict, not considering the 'sum' key
    num_sat = len(model_total[key]['Sat']['IntervalCounter']) - 1
    num_sun = len(model_total[key]['Sun']['IntervalCounter']) - 1

    if num_wd == 0:
        num_wd = 1
    
    if num_sat == 0:
        num_sat = 1

    if num_sun == 0:
        num_sun = 1

    model_total[key]['WD']["IntervalCounter"]['avg'] = list(map(lambda x: x/num_wd, model_total[key]['WD']["IntervalCounter"]['sum']))
    model_total[key]['WD']["IntervalCounter"]['std'] = list(map(lambda x: x*0.2, model_total[key]['WD']["IntervalCounter"]['avg']))

    model_total[key]['Sat']["IntervalCounter"]['avg'] = list(map(lambda x: x/num_sat, model_total[key]['Sat']["IntervalCounter"]['sum']))
    model_total[key]['Sat']["IntervalCounter"]['std'] = list(map(lambda x: x*0.2, model_total[key]['Sat']["IntervalCounter"]['avg']))

    model_total[key]['Sun']["IntervalCounter"]['avg'] = list(map(lambda x: x/num_sun, model_total[key]['Sun']["IntervalCounter"]['sum']))
    model_total[key]['Sun']["IntervalCounter"]['std'] = list(map(lambda x: x*0.2, model_total[key]['Sun']["IntervalCounter"]['avg']))

    for SA in model_total[key]['WD']["SourceAddress"].keys():
        num = len(model_total[key]['WD']["SourceAddress"][SA]) - 1 # removing sum key from the day logs
        model_total[key]['WD']["SourceAddress"][SA]['avg'] = model_total[key]['WD']["SourceAddress"][SA]['sum']/num
        model_total[key]['WD']["SourceAddress"][SA]['std'] = 0.2*model_total[key]['WD']["SourceAddress"][SA]['avg']

    for SA in model_total[key]['Sat']["SourceAddress"].keys():
        num = len(model_total[key]['Sat']["SourceAddress"][SA]) - 1
        model_total[key]['Sat']["SourceAddress"][SA]['avg'] = model_total[key]['Sat']["SourceAddress"][SA]['sum']/num
        model_total[key]['Sat']["SourceAddress"][SA]['std'] = 0.2*model_total[key]['Sat']["SourceAddress"][SA]['avg']

    for SA in model_total[key]['Sun']["SourceAddress"].keys():
        num = len(model_total[key]['Sun']["SourceAddress"][SA]) - 1
        model_total[key]['Sun']["SourceAddress"][SA]['avg'] = model_total[key]['Sun']["SourceAddress"][SA]['sum']/num
        model_total[key]['Sun']["SourceAddress"][SA]['std'] = 0.2*model_total[key]['Sun']["SourceAddress"][SA]['avg']

    for DH in model_total[key]['WD']["DestinationHost"].keys():
        num = len(model_total[key]['WD']["DestinationHost"][DH]) - 1 # removing sum key from the day logs
        model_total[key]['WD']["DestinationHost"][DH]['avg'] = model_total[key]['WD']["DestinationHost"][DH]['sum']/num
        model_total[key]['WD']["DestinationHost"][DH]['std'] = 0.2*model_total[key]['WD']["DestinationHost"][DH]['avg']

    for DH in model_total[key]['Sat']["DestinationHost"].keys():
        num = len(model_total[key]['Sat']["DestinationHost"][DH]) - 1
        model_total[key]['Sat']["DestinationHost"][DH]['avg'] = model_total[key]['Sat']["DestinationHost"][DH]['sum']/num
        model_total[key]['Sat']["DestinationHost"][DH]['std'] = 0.2*model_total[key]['Sat']["DestinationHost"][DH]['avg']

    for DH in model_total[key]['Sun']["DestinationHost"].keys():
        num = len(model_total[key]['Sun']["DestinationHost"][DH]) - 1
        model_total[key]['Sun']["DestinationHost"][DH]['avg'] = model_total[key]['Sun']["DestinationHost"][DH]['sum']/num
        model_total[key]['Sun']["DestinationHost"][DH]['std'] = 0.2*model_total[key]['Sun']["DestinationHost"][DH]['avg']

# Saving the data as json file, a lot of data 
with open("../models/TrainDataWeek_11.json", 'w') as f:
    json.dump(model_total, f)

f.close()

toc = process_time()
print("data logger saved")
# labelFile.close()
# csvFile.close()