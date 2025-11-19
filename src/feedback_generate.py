#### This code generates the feedback jsn files for all types of anomalies detected ####

import numpy as np
import csv
import json
from datetime import datetime, timedelta

# Utility functions
def getFeedback(intervals):

    user_feedback = {i: 'Negative' for i in intervals} # Dictionary with default negative i.e. none of the intervals have anomaly
    for i in intervals:
        if np.random.random() <= 0.6:  #60 percent positive data (correct anomalies)
            user_feedback[i] = 'Positive'

    return user_feedback #Dictionary

def getNewUserFeedback():

    # This is to get feedback from customer, whether a new user is anomalous or not
    if np.random.random() <= 0.25:  # 25 % are anomalous new users
        usr_feedback = 'New Positive'
    else:
        usr_feedback = 'New Negative'

    return usr_feedback

def getSourceFeedback():

    if np.random.random() <= 0.6:  # 60 % are correctly detected
        src_feedback = 'Positive'
    else:
        src_feedback = 'Negative'

    return src_feedback

def getDestinationFeedback():

    if np.random.random() <= 0.6:  # 60 % are correctly detected
        dest_feedback = 'Positive'
    else:
        dest_feedback = 'Negative'

    return dest_feedback

############### Writing feedback json for all anomalies with some probability ####################

def fb_generate(curr_date, time_anomaly, source_anomaly, dest_anomaly):

    np.random.seed(25)
    back_date = datetime.strptime("2023-06-29", "%Y-%m-%d")
    dictList = []
    with open("../outputs/UserFeedback.json", 'w') as f:
        for key in time_anomaly:

            feedback = {}
            feedback['DestinationUserName'] = key

            # if np.random.random() <= 0.5:
            feedback['StartDate'] = curr_date.strftime("%Y-%m-%d")
            # else:
            #     feedback['StartDate'] = back_date.strftime("%Y-%m-%d")

            # Obtaining feedback from customer for anomalous user 20% of the times
            if time_anomaly[key] != "New User":
                intervals = eval(time_anomaly[key].strip("Logon time, intervals: "))

                if np.random.random() <= 0.2:
                    feedback['Anomaly'] = getFeedback(intervals) # Dict with keys as the predicted anomalous intervals and the corresponding feedback

                else:
                    feedback['Anomaly'] = 'Nil'

            else:
                feedback['Anomaly'] = getNewUserFeedback()
                
            dictList.append(feedback)

        json.dump(dictList, f)
    f.close()

    ##### Source Feedback #####
    dictList = []
    with open("../outputs/SrcFeedback.json", 'w') as f:
        for key in source_anomaly:

            feedback = {}
            feedback['DestinationUserName'] = key
            # if np.random.random() <= 0.5:
            feedback['StartDate'] = curr_date.strftime("%Y-%m-%d")
            # else:
            #     feedback['StartDate'] = back_date.strftime("%Y-%m-%d")
            
            feedback['Anomaly'] = {}
            for SA in source_anomaly[key]:
                # Obtaining feedback from customer for anomalous source address 20% of the times
                if source_anomaly[key][SA] != "New Source Address":

                    if np.random.random() <= 0.2:
                        feedback['Anomaly'][SA] = getSourceFeedback() # Dict with keys as the source addresses and the corresponding feedback

                    else:
                        feedback['Anomaly'][SA] = 'Nil' # Feedback not obtained

                else:
                    feedback['Anomaly'][SA] = getNewUserFeedback() # New Source Address can be either anomalous or not
                
            dictList.append(feedback)

        json.dump(dictList, f)
    f.close()

    ##### Destination Feedback #####
    dictList = []
    with open("../outputs/DestFeedback.json", 'w') as f:
        for key in dest_anomaly:

            feedback = {}
            feedback['DestinationUserName'] = key
            # if np.random.random() <= 0.5:
            feedback['StartDate'] = curr_date.strftime("%Y-%m-%d")
            # else:
            #     feedback['StartDate'] = back_date.strftime("%Y-%m-%d")
            
            feedback['Anomaly'] = {}
            for DH in dest_anomaly[key]:
                # Obtaining feedback from customer for anomalous destination host 20% of the times
                if dest_anomaly[key][DH] != "New Destination Host":

                    if np.random.random() <= 0.2:
                        feedback['Anomaly'][DH] = getDestinationFeedback() # Dict with keys as the source addresses and the corresponding feedback

                    else:
                        feedback['Anomaly'][DH] = 'Nil' # Feedback not obtained

                else:
                    feedback['Anomaly'][DH] = getNewUserFeedback() # New Destination Host can be either anomalous or not
                
            dictList.append(feedback)

        json.dump(dictList, f)
    f.close()