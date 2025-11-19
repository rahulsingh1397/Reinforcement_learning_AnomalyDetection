# This code is to find a pattern in the orgfanization logons 
# during a weekday, sturday and sunday
# We will load our model and add all the summed interval counters for all users
# This way we will obtain three graphs

import numpy as np
import json
import matplotlib.pyplot as plt

# Load the json file
jsonName = "../models/TrainDataWeek_1.json"

with open(jsonName, 'r') as f:
    model_w1 = json.load(f)

# Load the json file
jsonName = "../models/TrainDataWeek_2.json"

with open(jsonName, 'r') as f:
    model_w2 = json.load(f)

# Load the json file
jsonName = "../models/saveTestData.json"
with open(jsonName, 'r') as f:
    model_test = json.load(f)

# Load anomalous users keys
jsonName = "../outputs/AnomalousUsers.json"
with open(jsonName, 'r') as f:
    model_anomaly = json.load(f)

# Load anomalous source addresses
jsonName = "../outputs/AnomalousSource.json"
with open(jsonName, 'r') as f:
    source_anomaly = json.load(f)

######## Processing and plotting ########
zero_list = np.zeros(8,)
user_model = {'WD': zero_list.copy(), 'Sat': zero_list.copy(), 'Sun': zero_list.copy()}

x_points = ['0-3', '3-6', '6-9', '9-12', '12-15', '15-18', '18-21', '21-24']

indx1 = np.arange(1075)
indx2 = np.arange(1085, len(model_w1))
indx = np.array(np.concatenate((indx1, indx2)))

key_list = list(model_w1.keys())
key_list2 = list(model_w2.keys())

# comparing common users in the two weeks, selecting random users
key_idx = np.random.randint(0, 100, 20)
common_keys = [key_list[i] for i in key_idx]
count = 0

# sum all the logons
for key in list(model_anomaly.keys()): #key_list2:

    if key in key_list and key in key_list2 and count < 10: #and model_anomaly[key] != "New user":

        count += 1
        # Week 1
        user_model['WD'] = np.array(model_w1[key]['WD']['IntervalCounter']['avg'])
        user_model['Sat'] = np.array(model_w1[key]['Sat']['IntervalCounter']['avg'])
        user_model['Sun'] = np.array(model_w1[key]['Sun']['IntervalCounter']['avg'])

        plt.figure()
        plt.plot(x_points, user_model['WD'], '-o', label = key + ": Weekday 1")
        plt.plot(x_points, user_model['Sat'], '-+', label = key + ": Saturday 1")
        plt.plot(x_points, user_model['Sun'], '-*', label = key + ": Sunday 1")

        # Week 2
        user_model['WD'] = np.array(model_w2[key]['WD']['IntervalCounter']['avg'])
        user_model['Sat'] = np.array(model_w2[key]['Sat']['IntervalCounter']['avg'])
        user_model['Sun'] = np.array(model_w2[key]['Sun']['IntervalCounter']['avg'])

        plt.plot(x_points, user_model['WD'], '-o', label = key + ": Weekday 2")
        plt.plot(x_points, user_model['Sat'], '-+', label = key + ": Saturday 2")
        plt.plot(x_points, user_model['Sun'], '-*', label = key + ": Sunday 2")

        ## Test day plot ##
        test_data = np.array(model_test[key]['WD']['IntervalCounter'])
        plt.plot(x_points, test_data, '--o', label = "Weekday (Friday): Test log")

        plt.ylabel("Average number of logons in a day by User")
        plt.xlabel("Intervals of 3 hours")
        plt.title("User behavior")

        plt.legend()   
        plt.show()

# Plotting the source/IP addresses anomaly plots for the test day
count = 0
for key in source_anomaly.keys():

    if key in key_list and key in key_list2 and count<= 10 and source_anomaly[key] != {}:

        count += 1
        plt.figure()

        for SA in source_anomaly[key]:
            if source_anomaly[key][SA] != "New Source Address":
                # Week 1
                user_model['WD'] = model_w1[key]['WD']['SourceAddress'][SA]['avg']

                plt.scatter(SA, user_model['WD'], c = 'r', label = 'Weekday 1')

                # Week 2
                # user_model['WD'] = model_w2[key]['WD']['SourceAddress'][SA]['avg']
                # plt.scatter(SA, user_model['WD'], c = 'g', label = 'Weekday 2')

                ## Test day plot ##
                test_data = model_test[key]['WD']['SourceAddress'][SA]
                plt.scatter(SA, test_data, c = 'b', label = 'Testday log')

        plt.ylabel("Average number of logons in a day by Source")
        plt.xlabel("Source Addresses")
        plt.title("User " + key + ": IP Address behavior")

        plt.legend()   
        plt.show()
