# this code is to merge all the data obtained from different log files into one model and write it to another csv
# for visualization

import numpy as np
import csv
import pickle
import json
# from dataPreprocessRawDict import*
from dataTestDictNew import*
from User_logon_anomaly_code import*
from feedback_generate import*
from datetime import datetime, timedelta
from time import process_time

# loading the entire model saved so far
state = input("Is the test model running for the first time? [Y/N]: ")

if state == 'Y':
    jsonFile = "../models/TrainDataWeek_1.json"
else:
    jsonFile = "../models/saveTrainDataUpdated2.json"

#################################### LOAD JSON FILES STARTS ###################################
# read the trained model json file
with open(jsonFile, 'r') as f:
    model_1 = json.load(f)

f.close()

with open("../models/TrainDataWeek_2.json", 'r') as f:
    model_2 = json.load(f)

f.close()

# loading the entire destination hostname list
destinationHosts = []
with open('../data/destinationlabel.csv', 'r') as destFile:
    next(destFile)
    reader = csv.reader(destFile, delimiter = ",")
    for row in reader:
        destinationHosts.append(row[1])

destFile.close()
#################################### LOAD JSON FILES ENDS ###################################

filePath = "../data/"
# curr_date = datetime.strptime("2023-07-04", "%Y-%m-%d")
# fileNames = ["SBM-2023-06-20.csv", "SBM-2023-06-24.csv", "SBM-2023-07-02.csv"]
model_current = {}
fileName = "SBM-2023-07-05/SBM-2023-07-04.csv"
curr_date = datetime.strptime("2023-07-04", "%Y-%m-%d")
num_logs = 100000
num_wd = 0
num_sat = 0
num_sun = 0
    
if curr_date.weekday() <=4:
    wd = 'WD'
elif curr_date.weekday() ==5:
    wd = 'Sat'
else:
    wd = 'Sun'

# day = -1

eof_flag = 0
tic = process_time()
# reading log data continuously
# for fileName in fileNames:
linecounter = 0
prev_interval = {}
threshold_val = [31, 69]

if state == 'Y':
    threshold_dict = {}
else:
    with open("../outputs/AnomalyThreshold.json", 'r') as f:
        threshold_dict = json.load(f)

# Obtaining the datalog fieldNames from json file
with open(filePath+fileName, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        field_names = list(row.keys())
        break

f.close()


# threshold_dict = {}
while not eof_flag:
# for i in range(1):

    # writing json file for the csv read, with each row as a dictionary
    writeJson(filePath+fileName, num_logs, linecounter, field_names)

    # model_current aggregate the cumulative data read so far i.e. not just one iteration but all of the data from starting
    model_current, destinationHosts, logsRead, eof_flag = writeToModel(model_current, destinationHosts, "../outputs/logData.json", curr_date, num_logs)
    linecounter += logsRead

    print("Number of logs read are: " + str(linecounter))
    # model_total, destinationHosts, day = writeToModel(model_total, destinationHosts, filePath+fileName, day)

    percent_criteria = 50 # 50 percent of sum of the average logon behaviour in a day
    anomaly = anomalyDetector(model_1, model_current, percent_criteria, threshold_val)

    # two different methods for "while reading file" and "reached to the eof"
    if not eof_flag:
        time_anomaly, prev_interval, threshold_dict = anomaly.logonTime_anomaly(prev_interval, wd, threshold_dict)
    else:
        time_anomaly, threshold_dict = anomaly.logonTime_eof_anomaly(wd, threshold_dict)

    print(time_anomaly)

    new_user_list = [key for key in time_anomaly.keys() if time_anomaly[key] == "New User"]

    # source address anomalies
    source_anomaly_1 = anomaly.source_anomaly(wd, eof_flag)

    # destination hosts anomalies
    dest_anomaly_1 = anomaly.dest_anomaly(wd, eof_flag)
    print(dest_anomaly_1)
    print("One Iteration completed")

    toc = process_time()

# Eliminating the empty key value pairs from original anomaly dictionary
source_anomaly = {}
dest_anomaly = {}
for key in source_anomaly_1:

    if source_anomaly_1[key] != {}:
        source_anomaly[key] = source_anomaly_1[key]

for key in dest_anomaly_1:

    if dest_anomaly_1[key] != {}:
        dest_anomaly[key] = dest_anomaly_1[key]


######### Generate feedbacks and write jsons for detected anomalies in user, source, destination
fb_generate(curr_date, time_anomaly, source_anomaly, dest_anomaly)
######### Complete ##########

##################### Writing JSON files for anomalies, thresholds and dest labels #####################
with open("../models/saveTestData.json", 'w') as f:
    json.dump(model_current, f)

f.close()

with open("../outputs/AnomalousUsers.json", 'w') as f:
    json.dump(time_anomaly, f)

f.close()

with open("../outputs/AnomalousSource.json", 'w') as f:
    json.dump(source_anomaly, f)

f.close()

with open("../outputs/AnomalousDestination.json", 'w') as f:
    json.dump(dest_anomaly, f)

f.close()

# updating destination label data (modifying the original file)
with open("../data/destinationLabel.csv", 'w', newline = '') as labelFile:
    labeldata = csv.writer(labelFile, delimiter = ",")
    labeldata.writerow(["Label", "Host Name"])
    for i in range(len(destinationHosts)):
        labeldata.writerow([i, destinationHosts[i]])

labelFile.close()

# Save the threshold dictionary for all the users
with open("../outputs/AnomalyThreshold.json", 'w') as f:
    json.dump(threshold_dict, f)
f.close()