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
    model_total = json.load(f)

# Load the json file
jsonName = "../models/TrainDataWeek_2.json"

with open(jsonName, 'r') as f:
    model_total2 = json.load(f)

zero_list = np.zeros(8,)
org_model = {'WD': zero_list.copy(), 'Sat': zero_list.copy(), 'Sun': zero_list.copy()}

x_points = ['0-3', '3-6', '6-9', '9-12', '12-15', '15-18', '18-21', '21-24']

indx1 = np.arange(1075)
indx2 = np.arange(1085, len(model_total))
indx = np.array(np.concatenate((indx1, indx2)))

key_list = list(model_total.keys())

# comparing common users in the two weeks, selecting random users
key_idx = np.random.randint(0, 100, 20)
common_keys = [key_list[i] for i in key_idx]

# number of days
num_wd = model_total[key_list[0]]['WD']["DayCounter"] #22
num_sat = model_total[key_list[0]]['Sat']["DayCounter"] #4
num_sun = model_total[key_list[0]]['Sun']["DayCounter"] #4

# sum all the logons
for key in key_list:

    org_model['WD'] += np.array(model_total[key]['WD']['IntervalCounter']['sum'])
    org_model['Sat'] += np.array(model_total[key]['Sat']['IntervalCounter']['sum'])
    org_model['Sun'] += np.array(model_total[key]['Sun']['IntervalCounter']['sum'])

    # plt.plot(x_points, np.array(model_total[key]['Sat']['IntervalCounter']['sum']), label = key)
    # plt.ylabel("Average number of logons in a day by User")
    # plt.xlabel("Intervals of 3 hours")
    # plt.title("User behavior on Saturdays")

# plt.legend()   
# plt.show()

org_model['WD'] /= num_wd
org_model['Sat'] /= num_sat
org_model['Sun'] /= num_sun

# plotting
plt.figure()
plt.plot(x_points, org_model['WD'], '-o', label = 'Week 1: Weekday')
plt.plot(x_points, org_model['Sat'], '-+', label = 'Week 1: Saturday')
plt.plot(x_points, org_model['Sun'], '-*', label = 'Week 1: Sunday')

plt.xlabel('Intervals of 3 hours')
plt.ylabel('Average Number of logons in a day')
plt.title("Organization Behaviour")
plt.legend()

############### Week 2 plot #################
org_model = {'WD': zero_list.copy(), 'Sat': zero_list.copy(), 'Sun': zero_list.copy()}
key_list = list(model_total2.keys())

# number of days
num_wd = model_total2[key_list[0]]['WD']["DayCounter"] #22
num_sat = model_total2[key_list[0]]['Sat']["DayCounter"] #4
num_sun = model_total2[key_list[0]]['Sun']["DayCounter"] #4

# sum all the logons
for key in key_list:

    org_model['WD'] += np.array(model_total2[key]['WD']['IntervalCounter']['sum'])
    org_model['Sat'] += np.array(model_total2[key]['Sat']['IntervalCounter']['sum'])
    org_model['Sun'] += np.array(model_total2[key]['Sun']['IntervalCounter']['sum'])


org_model['WD'] /= num_wd
org_model['Sat'] /= num_sat
org_model['Sun'] /= num_sun

# plotting
plt.plot(x_points, org_model['WD'], '-o', label = 'Week 2: Weekday')
plt.plot(x_points, org_model['Sat'], '-+', label = 'Week 2: Saturday')
plt.plot(x_points, org_model['Sun'], '-*', label = 'Week 2: Sunday')

plt.legend()
plt.show()

