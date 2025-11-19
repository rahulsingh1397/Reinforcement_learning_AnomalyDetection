# This code will implement a classic classificatrion problem
# based on given data for a user accessing the destination at different times of the day

# Importing all libraries
import numpy as np
from datetime import datetime

# import matplotlib.pyplot as plt

class anomalyDetector():

    def __init__(self, model_total, model, criteria, th_val):
        self.model_total = model_total
        self.model = model
        self.timeAnomaly = {}
        self.sourceAnomaly = {}
        self.destAnomaly = {}
        self.th = th_val
        self.mult_fac = 100/criteria

    def sig(self, x):

        out = 1/(1 + np.exp(-x))
        return out

    def logonTime_anomaly(self, prev_interval, dayType, anomaly_threshold):
        
        # For printing out alerts on anomaly intervals
        str_interval = ['0-3 hrs', '3-6 hrs', '6-9 hrs', '9-12 hrs', \
                        '12-15 hrs', '15-18 hrs', '18-21 hrs', '21-24 hrs']

        for key in self.model.keys():

            if key not in anomaly_threshold:
                anomaly_threshold[key] = self.th

            if key in self.model_total:
                x = np.array(self.model[key][dayType]["IntervalCounter"]) #['avg']
                avg = np.array(self.model_total[key][dayType]["IntervalCounter"]['avg'])
                sigma = np.array(self.model_total[key][dayType]["IntervalCounter"]['std'])
                curr_interval = self.model[key][dayType]["Interval"]

                num_wd = self.model_total[key][dayType]['DayCounter'] 
                # num_sat = self.model_total[key]['Sat']['DayCounter']
                # num_sun = self.model_total[key]['Sun']['DayCounter']

                if np.sum(avg) < 1.0:
                    avg_sum = 1.0

                else:
                    avg_sum = np.sum(avg)  # sum of average

                anomaly_int = (x <= avg + sigma)*(x >= avg - sigma)
                score = (x - avg)/(avg_sum)*self.mult_fac  # The score is defined as the ratio of difference between 
                # average value for the interval and the x value, divided by the sum of average values in all the intervals
                # We have considered a threshold of 50% equivalent to score of 1 (multiplication factor of 2)
                # When score = -+ 0.8 then the risk_score becomes 30.1% and 69.9% respectively

                # Risk score is defined as the sigmoid function acting on the above value to keep all the vals 
                # within 0 and 1
                risk_score = self.sig(score) * 100 # percentage, at score = 40 we get 50% and call it as anomalous
                
                ##### debugging ####
                # if key == "capital":
                #     pass

                if key not in prev_interval:
                    prev_interval[key] = curr_interval

                if curr_interval > prev_interval[key]:  
                
                    anomalous_intervals = np.where(np.logical_or(risk_score[0:-8+curr_interval] > anomaly_threshold[key][1], \
                                                risk_score[0:-8+curr_interval] < anomaly_threshold[key][0]))[0]
                    if len(anomalous_intervals) == 0:  
                    # if np.all(abs(risk_score[prev_interval[key]]) < 80): # looking at the completed interval
                        prev_interval[key] = curr_interval
                        # self.model_total[key]['WD']["IntervalCounter"]['avg'] = \
                        #        list(map(lambda x: x/num_wd, self.model_total[key]['WD']["IntervalCounter"]['sum']))
                    else: # get the indexes for the false values in anomaly
                        self.timeAnomaly[key] = "Logon time, intervals: " + str(list(anomalous_intervals)) #str(np.where(np.invert(anomaly_int))[0])
                        print("Anomaly found for user " + key + ": " + str([str_interval[i] for i in anomalous_intervals]))
                        # anomalous intervals
                        prev_interval[key] = curr_interval

                else:

                    if risk_score[curr_interval] <= anomaly_threshold[key][1]:
                        pass
                    else: # get the indexes for the false values in anomaly
                        self.timeAnomaly[key] = "Logon time, intervals: " + str(curr_interval) #str(np.where(np.invert(anomaly_int))[0])
                        print("Anomaly found for user " + key + ": " + str_interval[curr_interval])

                        
            else:
                self.timeAnomaly[key] = "New User"

        return self.timeAnomaly, prev_interval, anomaly_threshold

    # User logon time anomaly after the entire file/day logs are read (omitting prev and curr interval parts)
    def logonTime_eof_anomaly(self, dayType, anomaly_threshold):
        
        # For printing out alerts on anomaly intervals
        str_interval = ['0-3 hrs', '3-6 hrs', '6-9 hrs', '9-12 hrs', \
                        '12-15 hrs', '15-18 hrs', '18-21 hrs', '21-24 hrs']

        for key in self.model.keys():

            if key not in anomaly_threshold:
                anomaly_threshold[key] = self.th

            if key in self.model_total:
                x = np.array(self.model[key][dayType]["IntervalCounter"]) #['avg']
                avg = np.array(self.model_total[key][dayType]["IntervalCounter"]['avg'])
                sigma = np.array(self.model_total[key][dayType]["IntervalCounter"]['std'])
                curr_interval = self.model[key][dayType]["Interval"]

                num_wd = self.model_total[key][dayType]['DayCounter'] 
                # num_sat = self.model_total[key]['Sat']['DayCounter']
                # num_sun = self.model_total[key]['Sun']['DayCounter']

                if np.sum(avg) < 1.0:
                    avg_sum = 1.0

                else:
                    avg_sum = np.sum(avg)  # sum of average

                score = (x - avg)/(avg_sum)*self.mult_fac  # The score is defined as the ratio of difference between 
                # average value for the interval and the x value, divided by the sum of average values in all the intervals
                # We have considered a threshold of 50% equivalent to score of 1 (multiplication factor of 2)
                # When score = -+ 0.8 then the risk_score becomes 30.1% and 69.9% respectively

                # Risk score is defined as the sigmoid function acting on the above value to keep all the vals 
                # within 0 and 1
                risk_score = self.sig(score) * 100 # percentage, at score = 40 we get 50% and call it as anomalous
                
                # checking for all the intervals, if logon anomaly is present or not
                anomalous_intervals = np.where(np.logical_or(risk_score > anomaly_threshold[key][1], \
                                            risk_score < anomaly_threshold[key][0]))[0]
                if len(anomalous_intervals) == 0:  
                    pass

                else:
                    self.timeAnomaly[key] = "Logon time, intervals: " + str(list(anomalous_intervals)) #str(np.where(np.invert(anomaly_int))[0])
                    print("Anomaly found for user " + key + ": " + str([str_interval[i] for i in anomalous_intervals]))
                    # anomalous intervals

            else:
                self.timeAnomaly[key] = "New User"

        return self.timeAnomaly, anomaly_threshold


    def source_anomaly(self, dayType, eof):
        
        for key in self.model.keys():
            if key in self.model_total:

                self.sourceAnomaly[key] = {}
                avg_sum = np.sum(np.array(self.model_total[key][dayType]["IntervalCounter"]['avg'])) # Total number of logons per day

                if avg_sum < 1.0:
                    avg_sum = 1.0
                
                for SA in self.model[key][dayType]["SourceAddress"]:
                    if SA in self.model_total[key][dayType]["SourceAddress"]:  # main model
                        # check for the avg and sigma
                        x = np.array(self.model[key][dayType]["SourceAddress"][SA])
                        avg = np.array(self.model_total[key][dayType]["SourceAddress"][SA]['avg'])
                        sigma = np.array(self.model_total[key][dayType]["SourceAddress"][SA]['std'])

                        score = (x - avg)/(avg_sum)
                        risk_score = self.sig(score)*100  # risk is the percentage change in the x value with respect to the 
                        # average user behaviour based on trained model, 100% change equivalent to risk score of 100

                        # if x <= avg + sigma and x >= avg - sigma:
                        if not eof and risk_score <= 69:  # if the number of logons aren't exceedingly high i.e. 80 percent of the average
                            pass
                        elif eof and risk_score <= 69 and risk_score >= 31:
                            pass
                        else:
                            self.sourceAnomaly[key][SA] = "Source Address Anomaly " + str(risk_score)
                            print("Logon anomaly for " + key + " " + SA)

                    else: # not in main model i.e. first appearance
                        self.sourceAnomaly[key][SA] = "New Source Address"
                        print("New Source address found for " + key + " " + SA)

        return self.sourceAnomaly
    
    # Destination Host anomaly method
    def dest_anomaly(self, dayType, eof):

        for key in self.model.keys():
            if key in self.model_total:

                avg_sum = np.sum(np.array(self.model_total[key][dayType]["IntervalCounter"]['avg'])) # Total number of logons per day

                if avg_sum < 1.0:
                    avg_sum = 1.0

                self.destAnomaly[key] = {}

                for DH in self.model[key][dayType]["DestinationHost"]:
                    if str(DH) in self.model_total[key][dayType]["DestinationHost"]:
                        
                        # In model_total, DH is a string type whereas in self.model it is an integer
                        # This may be due to the fact that in model_total we have desthost label as a dictionary
                        # it is possible that the dictionary is represented as a string
                        x = np.array(self.model[key][dayType]["DestinationHost"][DH])
                        avg = np.array(self.model_total[key][dayType]["DestinationHost"][str(DH)]['avg'])
                        sigma = np.array(self.model_total[key][dayType]["DestinationHost"][str(DH)]['std'])

                        score = (x - avg)/(avg_sum)
                        risk_score = self.sig(score)*100  # risk is the percentage change in the x value with respect to the 
                            # average user behaviour based on trained model, 100 % change = risk score of 100

                        # if x <= avg + sigma and x >= avg - sigma:
                        if not eof and risk_score <= 69:  # if the number of logons aren't exceedingly high i.e. 80 percent of the average
                            pass
                        elif eof and risk_score <= 69 and risk_score >= 31:
                            pass
                        else:
                            self.destAnomaly[key][DH]= "Destination Host Anomaly " + str(risk_score)
                            print("Logon anomaly for " + key + " " + str(DH))

                    else: # not in main model i.e. first appearance
                        self.destAnomaly[key][DH]= "New Destination Host"
                        print("New Destination host found for " + key + " " + str(DH))

        return self.destAnomaly