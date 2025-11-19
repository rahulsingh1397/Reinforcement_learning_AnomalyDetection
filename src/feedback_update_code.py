# This code is to load all the models and the anomalous users predicted
# And then compare the predicted anomalies with the customer feedback
# If feedback is positive: Implies the anomaly detected is correct, no need to update the threshold or the value
# If feedback is negative: Implies the anomaly detected is incorrect, update the threshold by increasing it and take avergae of values
# Also update average values for other users

import numpy as np
import json
import copy
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

############################ LOAD JSON FILES #######################################
# Load the json file
# state = input("Is the feedback & update model running for the first time? [Y/N]: ")

# if state == 'Y':
#     jsonName = "TrainDataWeek_1.json"
# else:
#     jsonName = "saveTrainDataUpdated.json"

# with open(jsonName, 'r') as f:
#     model_w1 = json.load(f)

# # Load the json file
# jsonName = "saveTestData.json"
# with open(jsonName, 'r') as f:
#     model_test = json.load(f)

# # Load anomalous users keys
# # jsonName = "AnomalousUsers_old.json"
# # with open(jsonName, 'r') as f:
# #     model_anomaly = json.load(f)

# # Load anomalous users keys
# jsonName = "AnomalousUsers.json"
# with open(jsonName, 'r') as f:
#     model_anomaly3 = json.load(f)

# # for user in model_anomaly:

# #     if user not in model_anomaly3:
# #         print(user, model_anomaly[user])
# #         print(model_test[user]['WD']["IntervalCounter"])
# #         print(model_w1[user]['WD']["IntervalCounter"]['avg'])
# #         print(model_test[user]['WD']["Interval"])

# # Load anomalous source addresses
# jsonName = "AnomalousSource.json"
# with open(jsonName, 'r') as f:
#     source_anomaly = json.load(f)

# # Load anomalous destination hosts
# jsonName = "AnomalousDestination.json"
# with open(jsonName, 'r') as f:
#     dest_anomaly = json.load(f)

# # Load anomaly threshold for all the users
# jsonName = "AnomalyThreshold.json"
# with open(jsonName, 'r') as f:
#     threshold_dict = json.load(f)

#################################### LOAD JSON FILES ENDS ###################################

# utility functions: Sigmoid and model update for normal user behavior for future predictions
def sig(x):

    out = 1/(1+np.exp(-x))*100
    return out

def model_update(model_total, UN, dayType, x, avg_updated, counter, anomaly = None, model = None):

    # the elements of model are lists for model to be written in json dict format
    if anomaly == None:
        model_total[UN][dayType]["IntervalCounter"][counter] = list(x)
        model_total[UN][dayType]["IntervalCounter"]['sum'] += x
        model_total[UN][dayType]["IntervalCounter"]['sum'] = list(model_total[UN][dayType]["IntervalCounter"]['sum'])
        model_total[UN][dayType]["IntervalCounter"]['avg'] = list(avg_updated)
        model_total[UN][dayType]["IntervalCounter"]['std'] = list(0.2*avg_updated)

    elif anomaly == "New User":
        model_total[UN] = {}
        model_total[UN]["UserLabel"] = len(model_total)
        model_total[UN]['WD'] = {"DayCounter": 0, "IntervalCounter": {0:[0,0,0,0,0,0,0,0], 'sum':[0,0,0,0,0,0,0,0]}, \
                                  "SourceAddress": {}, "DestinationHost": {}}
        model_total[UN]['Sat'] = {"DayCounter": 0, "IntervalCounter": {0:[0,0,0,0,0,0,0,0], 'sum':[0,0,0,0,0,0,0,0]}, \
                                  "SourceAddress": {}, "DestinationHost": {}}
        model_total[UN]['Sun'] = {"DayCounter": 0, "IntervalCounter": {0:[0,0,0,0,0,0,0,0], 'sum':[0,0,0,0,0,0,0,0]}, \
                                  "SourceAddress": {}, "DestinationHost": {}}

        model_total[UN][dayType]["IntervalCounter"][0] = list(x)
        model_total[UN][dayType]["IntervalCounter"]['sum'] += x
        model_total[UN][dayType]["IntervalCounter"]['sum'] = list(model_total[UN][dayType]["IntervalCounter"]['sum'])
        model_total[UN][dayType]["IntervalCounter"]['avg'] = list(avg_updated)
        model_total[UN][dayType]["IntervalCounter"]['std'] = list(0.2*avg_updated)

        # update source address and dest host dictionaries
        for SA in model["SourceAddress"]:
            model_total[UN][dayType]["SourceAddress"][SA] = {0: model["SourceAddress"][SA], 'sum': model["SourceAddress"][SA], \
                                                             'avg': model["SourceAddress"][SA], 'std': 0.2*model["SourceAddress"][SA]}

        for DH in model["DestinationHost"]:
            model_total[UN][dayType]["DestinationHost"][DH] = {0: model["DestinationHost"][DH], 'sum': model["DestinationHost"][DH], \
                                                             'avg': model["DestinationHost"][DH], 'std': 0.2*model["DestinationHost"][DH]}

    else: # update model in response to negative feedback for anomalies
        # x_updated = np.array([0 if a == -1 else a for a in x ]) # so that only updated intervals are changed, rest remains same
        model_total[UN][dayType]["IntervalCounter"][counter] = list(x)

        # model is actually update intervals, for this case. So, we will update sum only for update_interval
        for i in model:
            model_total[UN][dayType]["IntervalCounter"]['sum'][i] += x[i]
        model_total[UN][dayType]["IntervalCounter"]['sum'] = list(model_total[UN][dayType]["IntervalCounter"]['sum'])
        model_total[UN][dayType]["IntervalCounter"]['avg'] = list(avg_updated)
        model_total[UN][dayType]["IntervalCounter"]['std'] = list(0.2*avg_updated)

    # return model_total

# source address dictionary update
def source_update(model_total, UN, dayType, SA, x, avg_updated, counter, anomaly = None):

    # the elements of model are lists for model to be written in json dict format
    if anomaly == "New SA":

        model_total[UN][dayType]["SourceAddress"][SA] = {counter: x}
        model_total[UN][dayType]["SourceAddress"][SA]['sum'] = x
        model_total[UN][dayType]["SourceAddress"][SA]['avg'] = avg_updated
        model_total[UN][dayType]["SourceAddress"][SA]['std'] = 0.2*avg_updated

    elif anomaly == "No Feedback":

        model_total[UN][dayType]["SourceAddress"][SA][counter] = x
        model_total[UN][dayType]["SourceAddress"][SA]['avg'] = avg_updated
        model_total[UN][dayType]["SourceAddress"][SA]['std'] = 0.2*avg_updated

    else: # update model in response to negative feedback for anomalies as well as no anomaly
        model_total[UN][dayType]["SourceAddress"][SA][counter] = x
        model_total[UN][dayType]["SourceAddress"][SA]['sum'] += x
        model_total[UN][dayType]["SourceAddress"][SA]['avg'] = avg_updated
        model_total[UN][dayType]["SourceAddress"][SA]['std'] = 0.2*avg_updated

# Destination host dictionary update
def destination_update(model_total, UN, dayType, DH, x, avg_updated, counter, anomaly = None):

    if anomaly == "New DH":
        model_total[UN][dayType]["DestinationHost"][DH] = {counter: x}
        model_total[UN][dayType]["DestinationHost"][DH]['sum'] = x
        model_total[UN][dayType]["DestinationHost"][DH]['avg'] = avg_updated
        model_total[UN][dayType]["DestinationHost"][DH]['std'] = 0.2*avg_updated

    elif anomaly == "No Feedback": # storing data and not updating sum or avg
        model_total[UN][dayType]["DestinationHost"][DH][counter] = x
        model_total[UN][dayType]["DestinationHost"][DH]['avg'] = avg_updated
        model_total[UN][dayType]["DestinationHost"][DH]['std'] = 0.2*avg_updated

    else: # update model in response to negative feedback for anomalies as well as no anomaly
        model_total[UN][dayType]["DestinationHost"][DH][counter] = x
        model_total[UN][dayType]["DestinationHost"][DH]['sum'] += x
        model_total[UN][dayType]["DestinationHost"][DH]['avg'] = avg_updated
        model_total[UN][dayType]["DestinationHost"][DH]['std'] = 0.2*avg_updated

def getDayType(date):

    if date.weekday() <= 4:
        dayType = 'WD'
    elif date.weekday() == 5:
        dayType = 'Sat'
    else:
        dayType = 'Sun'

    return dayType

##################### Utility functions end ########################

# Model update function with input arguments from a main function
# There are 2 outputs, the updated trained model and anomaly thresholds if needed

def trainModelUpdate(date, model_w1, model_test, model_anomaly3, source_anomaly, dest_anomaly, threshold_dict):

    np.random.seed(25)

    # Type as in WD, Sat or Sun
    dayType = getDayType(date)
        
    # Load feedback data from userFeedback json or create your own data for all the users
    try:
        with open("../outputs/UserFeedback.json", 'r') as f:
            feedback = json.load(f)
        f.close()

        with open("../outputs/SrcFeedback.json", 'r') as f:
            src_feedback = json.load(f)
        f.close()

        with open("../outputs/DestFeedback.json", 'r') as f:
            dest_feedback = json.load(f)
        f.close()

        # for iter in fb:
        #     name = iter.pop("DestinationUserName")
        #     feedback[name] = iter
    
    except:
        feedback = {}
        src_feedback = {}
        dest_ffedback = {}

    ######################################################
    x_points = ['0-3', '3-6', '6-9', '9-12', '12-15', '15-18', '18-21', '21-24']
    mult_fac = 2

    # updating the threshold values and the average values for incorrect user anomaly prediction
    model_new = copy.deepcopy(model_w1)
    threshold_updated = 0
    anomaly_not_detected = 0
    num_new = 0

    # define a feedback received flag dictionary for each user which is: 0 if no feedback is received, 1 if a feedback is received
    feedback_flag = {}

    for fb in feedback: # Loop over all messages in the feedback dict

        key = fb['DestinationUserName']
        if key == 'AH000985':
            pass
        
        if 'New' not in fb['Anomaly']: # Not for new users

            # we will update the average for those intervals which are not anomalous based on feedback
            x = np.array(model_test[key][dayType]["IntervalCounter"], dtype = np.float64)
            avg = np.array(model_w1[key][dayType]["IntervalCounter"]['avg'])
            
            # original scores
            if np.sum(avg) < 1:
                avg_sum = 1

            else:
                avg_sum = np.sum(avg)

            score = (x - avg)/avg_sum*mult_fac
            risk_score = sig(score)
            # determine intervals of anomalous behavior
            anomalous_intervals = np.where(np.logical_or(risk_score > threshold_dict[key][1], \
                                                    risk_score < threshold_dict[key][0]))[0]

            # a dictionary storing the dates when no feedback was obtained and the corresponding dayCount
            # As and when feedback is obtained for that day the date is popped out of the dictionary
            if "NoFeedback" not in model_new[key][dayType]["IntervalCounter"]: # First time anomaly encountered
                model_new[key][dayType]["IntervalCounter"]["NoFeedback"] = {}

            feedback_date = fb['StartDate']
            newdayType = getDayType(datetime.strptime(feedback_date, "%Y-%m-%d"))

            if fb['Anomaly'] != 'Nil' and date.strftime("%Y-%m-%d") == feedback_date: #dayType is also same as newdayType
                
                not_update = [int(index) for index, v in fb['Anomaly'].items() if v == 'Positive']
                update_interval = [i for i in range(8) if i not in not_update]
                # The anomalous intervals are represented as -1
                x_new = x.copy()
                x_new[not_update] = -1  # Positive feedback intervals to be -1, rest all same as x

                dayCount = model_w1[key][dayType]['DayCounter']
                counter = len(model_w1[key][dayType]['IntervalCounter']) - 3 # Number of different days logged so far (Avg calc.)
                
            elif fb['Anomaly'] != 'Nil' and date.strftime("%Y-%m-%d") != feedback_date:

                # Two updations required, one for the feedback date and one for today
                not_update = [int(index) for index, v in fb['Anomaly'].items() if v == 'Positive']
                update_interval = [i for i in range(8) if i not in not_update]
                # Number of different days logged up to the feedback date (Avg calc.)
                counter = len(model_w1[key][newdayType]['IntervalCounter']) - \
                                                len(model_w1[key][newdayType]["IntervalCounter"]["NoFeedback"]) - 3
                # extracting the daycounter corresponding to feedback date (model_new)
                dayCount = model_new[key][newdayType]["IntervalCounter"]["NoFeedback"].pop(feedback_date)
                x_new = np.array(model_w1[key][newdayType]["IntervalCounter"][dayCount])
                # Only negative feedback are updated, rest need not be ('Positive feedback')
                x_new[not_update] = -1

                # feedback_flag[key] = 1  # This key has occurred once

            else: # No feedback ('Nil') obtained for the present day
                
                x_new = x # if no feedback comes we will store the original x to be used for future
                #No need to update the average and sum for any of the intervals
                dayCount = model_w1[key][dayType]['DayCounter']
                model_new[key][dayType]["IntervalCounter"]["NoFeedback"][date.strftime("%Y-%m-%d")] = dayCount
                counter = len(model_w1[key][dayType]['IntervalCounter']) - 3
                update_interval = []

            #'Positive': # Anomaly detected correct
            #'Negative': Anomaly detected incorrectly or the logon behavior is correct and can be used to update average
            # storing the values only which are to be updated (auxilliary variables)
            xi = x[update_interval]
            avgi = avg[update_interval]
            # New avergae is calculated based on the total number of days logged so far and not just the mean of two values
            avg_new = avg.copy()
            avg_new[update_interval] = np.round((xi + avgi*counter)/(counter + 1), decimals = 2)

            # new scores
            if np.sum(avg_new) < 1:
                avg_new_sum = 1

            else:
                avg_new_sum = np.sum(avg_new)

            new_score =  (x - avg_new)/avg_new_sum*mult_fac
            new_risk_score = sig(new_score)

            # plt.figure()
            # plt.plot(x_points, risk_score, '-o', label = 'Risk Score previous')
            # plt.plot(x_points, new_risk_score, '-+', label = 'Risk Score new')
            # plt.title("User " + key + " risk scores")
            # plt.xlabel("Intervals of 3 hours")
            # plt.ylabel("Risk Score (0-100)")
            # plt.legend()
            # plt.show()

            # update the original model with new data
            model_update(model_new, key, newdayType, x_new, avg_new, dayCount, 'Logon Anomaly', update_interval)

            # updating risk score threshold for anomaly detection, for the Negative intervals only
            if fb['Anomaly'] != 'Nil':
                if min(new_risk_score[update_interval]) < threshold_dict[key][0]:
                        threshold_dict[key][0] = (min(new_risk_score[update_interval]) + threshold_dict[key][0])/2
                        # average of the previous threshold and new risk score to update threshold slightly
                        threshold_updated += 1

                if max(new_risk_score[update_interval]) > threshold_dict[key][1]:
                        threshold_dict[key][1] = (max(new_risk_score[update_interval]) + threshold_dict[key][1])/2
                        # average of the previous threshold and new risk score to update threshold slightly
                        threshold_updated += 1

        else: # New Users

            num_new += 1
            x = np.array(model_test[key][dayType]["IntervalCounter"], dtype = np.float64)
            avg_new = x
            dayCount = 0

            # usr_feedback = feedback[key]['Anomaly']

            # if usr_feedback == 'New Positive': # Anomalous new user found as per customer
            #     pass
            # else: # new user normal behavior
                # update the original model with new data
            model_update(model_new, key, dayType, x, avg_new, dayCount, 'New User', model_test[key][dayType])

    ################# Update the non anomalous users #################
    for key in model_test:

        if key not in model_anomaly3:  # Non anomalous users

            x = np.array(model_test[key][dayType]["IntervalCounter"], dtype = np.float64)
            avg = np.array(model_w1[key][dayType]["IntervalCounter"]['avg'])
            counter = len(model_w1[key][dayType]['IntervalCounter']) - 3
            dayCount = model_w1[key][dayType]["DayCounter"]

            # New average is calculated based on the total number of days logged so far and not just the mean of two values
            avg_new = np.round((x + avg*counter)/(counter + 1), decimals = 2)

            if np.sum(avg_new) < 1:
                print(key)
            new_score =  (x - avg_new)/np.sum(avg_new)*mult_fac
            new_risk_score = sig(new_score)

            # update the original model with new data
            model_update(model_new, key, dayType, x, avg_new, dayCount)

            # updating risk score threshold for anomaly detection
            # Note: since the new average is closer to x, the risk score is supposed to get closer to the 
            # threshold range and thus it will never exceed the range for this case
            if min(new_risk_score) < threshold_dict[key][0]:
                    threshold_dict[key][0] = min(new_risk_score)

            if max(new_risk_score) > threshold_dict[key][1]:
                    threshold_dict[key][1] = max(new_risk_score)
                    anomaly_not_detected += 1

    ##### Upadting daycounter to += 1
    for key in model_new.keys():
        model_new[key][dayType]['DayCounter'] += 1
    ####################### updating user logons for each interval completed ############################

    ####################### Updating source addresses #######################
    for src in src_feedback:

        key = src['DestinationUserName']
        feedback_date = src['StartDate']
        newdayType = getDayType(datetime.strptime(feedback_date, "%Y-%m-%d"))
            
        for SA in src['Anomaly']:
            
            if 'New' in src['Anomaly'][SA]:
                # Note: If SA is seen for the first time, its average is taken based on the days it is used i.e. 1
                counter = 0
                logons = model_test[key][newdayType]["SourceAddress"][SA]
                avg_new = round(logons/(counter + 1), 2)
                dayCount = model_w1[key][newdayType]['DayCounter']

                source_update(model_new, key, newdayType, SA, logons, avg_new, dayCount, "New SA")

            elif 'New' not in src['Anomaly'][SA]:

                if "NoFeedback" not in model_new[key][dayType]["SourceAddress"][SA]:
                    model_new[key][dayType]["SourceAddress"][SA]["NoFeedback"] = {} # For existing source addresses only
                # Since nofeedback will have data only it has occurred previously

                # source_feedback = getSourceFeedback()
                if src['Anomaly'][SA] == 'Positive':
                    pass

                elif date.strftime("%Y-%m-%d") == feedback_date and src['Anomaly'][SA] == 'Negative': # update the source address logon counts
                    # newdayType and dayType are the same for this case
                    counter = len(model_w1[key][newdayType]["SourceAddress"][SA]) - 3 #not considering sum, avg and std keys in SA
                    logons = model_test[key][newdayType]["SourceAddress"][SA]
                    avg = model_w1[key][newdayType]["SourceAddress"][SA]['avg']
                    avg_new = round((logons + counter*avg)/(counter + 1), 2)
                    dayCount = model_w1[key][newdayType]['DayCounter']

                    source_update(model_new, key, newdayType, SA, logons, avg_new, dayCount, "Logon Anomaly")

                elif date.strftime("%Y-%m-%d") != feedback_date and src['Anomaly'][SA] == 'Negative': # update the source address logon counts
                    counter = len(model_w1[key][newdayType]["SourceAddress"][SA]) - \
                              len(model_w1[key][newdayType]["SourceAddress"][SA]["NoFeedback"]) - 3 #not considering sum, avg and std keys in SA
                    
                    # We have to pop out from new model
                    dayCount = model_new[key][newdayType]["SourceAddress"][SA]["NoFeedback"].pop(feedback_date)
                    logons = model_w1[key][newdayType]["SourceAddress"][SA][dayCount]
                    avg = model_w1[key][newdayType]["SourceAddress"][SA]['avg']
                    avg_new = round((logons + counter*avg)/(counter + 1), 2)

                    source_update(model_new, key, newdayType, SA, logons, avg_new, dayCount, "Logon Anomaly")

                else:  # No feedback obtained 'Nil'
                    logons = model_test[key][dayType]["SourceAddress"][SA]
                    dayCount = model_w1[key][dayType]['DayCounter']
                    avg = model_w1[key][dayType]["SourceAddress"][SA]['avg']
                    # Add the No feedback data
                    model_new[key][dayType]["SourceAddress"][SA]["NoFeedback"][feedback_date] = dayCount

                    source_update(model_new, key, dayType, SA, logons, avg, dayCount, "No Feedback")

    # update the SA with no anomaly and the users which have no source anomaly
    for key in model_test: 
        
        if key in source_anomaly:
            for SA in model_test[key][dayType]["SourceAddress"]:
                
                if SA not in source_anomaly[key]: # SA with no anomaly detected (today) in the key, need to be updated

                    counter = len(model_w1[key][dayType]["SourceAddress"][SA]) - 3
                    logons = model_test[key][dayType]["SourceAddress"][SA]
                    avg = model_w1[key][dayType]["SourceAddress"][SA]['avg']
                    avg_new = round((logons + counter*avg)/(counter + 1), 2)
                    dayCount = model_w1[key][dayType]['DayCounter']

                    source_update(model_new, key, dayType, SA, logons, avg_new, dayCount)
    
        elif key in model_w1:  # key not found in source anomaly and not a New User, need to updated for all source addresses
        # New User source addresses have been updated already

            for SA in model_test[key][dayType]["SourceAddress"]:
                counter = len(model_w1[key][dayType]["SourceAddress"][SA]) - 3
                logons = model_test[key][dayType]["SourceAddress"][SA]
                avg = model_w1[key][dayType]["SourceAddress"][SA]['avg']
                avg_new = round((logons + counter*avg)/(counter + 1), 2)
                dayCount = model_w1[key][dayType]['DayCounter']

                source_update(model_new, key, dayType, SA, logons, avg_new, dayCount)
    ################## Updating Source Address dictionary completed ###################

    ################## Updating destination host names dictionary #####################
    for dest in dest_feedback:

        key = dest["DestinationUserName"]
        feedback_date = dest['StartDate']
        newdayType = getDayType(datetime.strptime(feedback_date, "%Y-%m-%d"))

        for DH in dest['Anomaly']:
            
            if 'New' in dest['Anomaly'][DH]: # New DH need to updated as a new dict
                    
                    # Since DH is seen for the first time, its average will be same as the recorded value i.e. counter = 0
                    counter = 0
                    logons = model_test[key][dayType]["DestinationHost"][DH]
                    avg_new = round((logons)/(counter + 1), 2)
                    dayCount = model_w1[key][dayType]['DayCounter']

                    destination_update(model_new, key, dayType, DH, logons, avg_new, dayCount, "New DH")

            elif 'New' not in dest['Anomaly'][DH] :

                if "NoFeedback" not in model_new[key][dayType]["DestinationHost"][DH]: # First encounter
                    model_new[key][dayType]["DestinationHost"][DH]["NoFeedback"] = {} # Existing destination host names
                    # can only be assigned a new key NoFeedback

                if dest['Anomaly'][DH] == 'Positive': # correctly detected anomaly is ignored for updation
                    pass

                elif date.strftime("%Y-%m-%d") == feedback_date and dest['Anomaly'][DH] == 'Negative':
                    # newdayType same as dayType
                    counter = len(model_w1[key][dayType]["DestinationHost"][DH]) - 3 # ignoring sum, avg and std keys
                    logons = model_test[key][dayType]["DestinationHost"][DH]
                    avg = model_w1[key][dayType]["DestinationHost"][DH]['avg']
                    avg_new = round((logons + avg*counter)/(counter + 1), 2)
                    dayCount = model_w1[key][dayType]['DayCounter']

                    destination_update(model_new, key, dayType, DH, logons, avg_new, dayCount, "Logon Anomaly")
                
                elif date.strftime("%Y-%m-%d") != feedback_date and dest['Anomaly'][DH] == 'Negative':
                    # newdayType same as dayType
                    counter = len(model_w1[key][dayType]["DestinationHost"][DH]) - \
                              len(model_w1[key][dayType]["DestinationHost"][DH]["NoFeedback"]) - 3 # ignoring sum, avg and std keys
                    
                    # pop out from the new model so that it is updated
                    dayCount = model_new[key][dayType]["DestinationHost"][DH]["NoFeedback"].pop()
                    logons = model_w1[key][dayType]["DestinationHost"][DH][dayCount]
                    avg = model_w1[key][dayType]["DestinationHost"][DH]['avg']
                    avg_new = round((logons + avg*counter)/(counter + 1), 2)

                    destination_update(model_new, key, dayType, DH, logons, avg_new, dayCount, "Logon Anomaly")

                else: # For No feedback obtained ('Nil')
                    logons = model_test[key][dayType]["DestinationHost"][DH]
                    dayCount = model_w1[key][dayType]['DayCounter']
                    avg = model_w1[key][dayType]["DestinationHost"][DH]['avg']
                    # Add the No feedback data
                    model_new[key][dayType]["DestinationHost"][DH]["NoFeedback"][feedback_date] = dayCount

                    destination_update(model_new, key, dayType, DH, logons, avg, dayCount, "No Feedback")

    # update the DH with no anomaly and the users/key which have no destination anomaly
    for key in model_test: 
        
        if key in dest_anomaly:
            for DH in model_test[key][dayType]["DestinationHost"]: # DH is integer
                
                if DH not in dest_anomaly[key]: # DH not in dest_anomaly should be updated

                    str_DH = str(DH)
                    counter = len(model_w1[key][dayType]["DestinationHost"][str_DH]) - 3
                    logons = model_test[key][dayType]["DestinationHost"][DH]
                    avg = model_w1[key][dayType]["DestinationHost"][str_DH]['avg']
                    avg_new = round((logons + avg*counter)/(counter + 1), 2)
                    dayCount = model_w1[key][dayType]['DayCounter']

                    destination_update(model_new, key, dayType, str_DH, logons, avg_new, dayCount)

        elif key in model_w1: # Update the keys which are neither anomalous nor New Users i.e. exist in model_w1 but not in dest_anomaly

            for DH in model_test[key][dayType]["DestinationHost"]:
                str_DH = str(DH)
                counter = len(model_w1[key][dayType]["DestinationHost"][str_DH]) - 3
                logons = model_test[key][dayType]["DestinationHost"][DH]
                avg = model_w1[key][dayType]["DestinationHost"][str_DH]['avg']
                avg_new = round((logons + avg*counter)/(counter + 1), 2)
                dayCount = model_w1[key][dayType]['DayCounter']

                destination_update(model_new, key, dayType, str_DH, logons, avg_new, dayCount)

    print("Updation completed as per my understanding")

    return model_new, threshold_dict
