# This code is for learning occ implementation for a synthetic sklearn data

# importing all the necessary libraries
import numpy as np
import pandas as pd
import csv
from datetime import datetime
from time import process_time

# Synthetic datasets

##########################################
# methods
def convert2list(str):
     li = list(str.split(" "))
     return li

def IP2dec(ip):
     # dec = np.zeros(len(iplist))
     # for ip in iplist:
     ip_split = list(ip.split("."))  # converting an ip into list of 4 octets
     ip_int = list(map(int, ip_split))
     dec = ip_int[0]*256**3 + ip_int[1]*256**2 + ip_int[2]*256 + ip_int[3]  # element wise

     return dec

def addToModel(curr_model, logdata, wday, interval):

     # wday is the weekday/Sat/Sun obtained from weekday method of datetime
     UN = logdata["UserName"]
     if  UN in curr_model:
          # append is a non returnable method
          # Now we will be storing data for each day
          # curr_model[UN]["StartDate"].append(logdata["StartDate"])
          counter = curr_model[UN][wday]["DayCounter"]  # day counter for weekday, sat or sun: Starts with 0

          # counter is the day ##############

          if counter in curr_model[UN][wday]["IntervalCounter"]:
               curr_model[UN][wday]["IntervalCounter"][counter][interval] += 1

          else:
               curr_model[UN][wday]["IntervalCounter"][counter] = [0,0,0,0,0,0,0,0]
               curr_model[UN][wday]["IntervalCounter"][counter][interval] += 1

          curr_model[UN][wday]["IntervalCounter"]['sum'][interval] += 1
          # curr_model[UN]["EndDate"].append(logdata["EndDate"])
          
          # Sources
          if logdata["SourceAddress"] not in curr_model[UN][wday]["SourceAddress"]:
               curr_model[UN][wday]["SourceAddress"][logdata["SourceAddress"]] = {counter: 1}  # Dict created
               curr_model[UN][wday]["SourceAddress"][logdata["SourceAddress"]]['sum'] = 1

          elif counter not in curr_model[UN][wday]["SourceAddress"][logdata["SourceAddress"]]:
               curr_model[UN][wday]["SourceAddress"][logdata["SourceAddress"]][counter] = 1  # dictionary already existed
               curr_model[UN][wday]["SourceAddress"][logdata["SourceAddress"]]['sum'] += 1

          else:
               curr_model[UN][wday]["SourceAddress"][logdata["SourceAddress"]][counter] += 1
               curr_model[UN][wday]["SourceAddress"][logdata["SourceAddress"]]['sum'] += 1
          
          # Destinations
          if logdata["DestinationHost"] not in curr_model[UN][wday]["DestinationHost"]:
               curr_model[UN][wday]["DestinationHost"][logdata["DestinationHost"]] = {counter: 1}
               curr_model[UN][wday]["DestinationHost"][logdata["DestinationHost"]]['sum'] = 1
          
          elif counter not in curr_model[UN][wday]["DestinationHost"][logdata["DestinationHost"]]:
               curr_model[UN][wday]["DestinationHost"][logdata["DestinationHost"]][counter] = 1
               curr_model[UN][wday]["DestinationHost"][logdata["DestinationHost"]]['sum'] += 1

          else:
               curr_model[UN][wday]["DestinationHost"][logdata["DestinationHost"]][counter] += 1
               curr_model[UN][wday]["DestinationHost"][logdata["DestinationHost"]]['sum'] += 1
    
     else:
          curr_model[UN] = {}
          
          # [0,0,0,0,0,0,0,0]
          # For each wday we have Int counter, SA and DH names
          curr_model[UN]['WD'] = {"DayCounter": 0, "IntervalCounter": {0:[0,0,0,0,0,0,0,0], 'sum':[0,0,0,0,0,0,0,0]}, \
                                  "SourceAddress": {}, "DestinationHost": {}}
          curr_model[UN]['Sat'] = {"DayCounter": 0, "IntervalCounter": {0:[0,0,0,0,0,0,0,0], 'sum':[0,0,0,0,0,0,0,0]}, \
                                  "SourceAddress": {}, "DestinationHost": {}}
          curr_model[UN]['Sun'] = {"DayCounter": 0, "IntervalCounter": {0:[0,0,0,0,0,0,0,0], 'sum':[0,0,0,0,0,0,0,0]}, \
                                  "SourceAddress": {}, "DestinationHost": {}}
          ######################################

          curr_model[UN]["UserLabel"] = len(curr_model)  # As soon as we specify a dict key (UN) its size increases by 1
          # curr_model[UN]["StartDate"] = list([logdata["StartDate"]])  # list of string timestamps
          curr_model[UN][wday]["IntervalCounter"][0][interval] += 1
          curr_model[UN][wday]["IntervalCounter"]['sum'][interval] += 1
          curr_model[UN][wday]["SourceAddress"] = {logdata["SourceAddress"]: {0:1, 'sum':1}} # 
          curr_model[UN][wday]["DestinationHost"] = {logdata["DestinationHost"]: {0:1, 'sum':1}} # list of 1D arrays

     return curr_model    

def writeToModel(model, destination, filePathName, interval = 3):

     with open(filePathName, 'r') as file:
          next(file) # skipping header line
          reader = csv.reader(file, delimiter = ",")
          linecounter = 0
          # wday = None
          # destination = []
          for row in reader:
               if linecounter < 500000:
                    linecounter += 1
                    record = {}
                    try:
                         logonType = int(row[12])  # FirstLogin Device Custom Number1
                         # startDate = datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S")
                         Name = row[3]

                         if "logged on" in Name and logonType in [2,3,7,9,10]:

                              # Record the data
                              startDate = row[0]  # timestamp in UTC
                              Date = datetime.utcfromtimestamp(int(row[0])/1000)
                              startDate = Date.strftime("%Y-%m-%d %H:%M:%S")

                              interval = int(Date.hour/3) # 24 hrs is 00:00 (0 to 7) [0,3), [3,6) ...

                              # if wday is None: # required to be done just once in a one day log
                              if Date.weekday() <=4: # Mon to Fri
                                   wday = 'WD'

                              elif Date.weekday() == 5: # Sat
                                   wday = 'Sat'

                              else:
                                   wday = 'Sun'
                              
                              # endDate = row[5]

                              sourceAddress = row[4]
                              # sourceAddressList = convert2list(sourceAddress) # list of IPs
                              if len(sourceAddress) == 0:
                                   source = 'NIL'
                              
                              else:
                                   # source = IP2dec(sourceAddress)
                                   source = sourceAddress

                              userName = row[9]

                              destinationHostName = row[8]  # Just one name
                              # destinationHostNameList =  convert2list(destinationHostName)

                              if destinationHostName in destination:
                                   destinationLabel = destination.index(destinationHostName)

                              else:
                                   destination.append(destinationHostName)  # it is a method with no return and appends the list with each
                              # element of the added list
                                   destinationLabel = len(destination)-1
                              
                              if linecounter % 100000 == 0:
                                   print(linecounter/ 100000)

                              record["UserName"] = userName
                              record["StartDate"] = startDate
                              # record["EndDate"] = endDate
                              record["SourceAddress"] = source
                              record["DestinationHost"] = destinationLabel
                              # record["DestinationHostName"] = destinationHostNameList

                              model = addToModel(model, record, wday, interval)
                              print("Recorded data added to model", linecounter)
               
                    except Exception as ex:
                         pass

               else:
                    print("500,000 data logs have been read")
                    break

     file.close()
     return model, destination

##################################################

# str = ["abc", "bcd", "839", "abc", "839", "aac"]
# tic = process_time()
# print(list(dict.fromkeys(str)))
# # print(np.unique(str))
# toc = process_time()