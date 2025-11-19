# This code is for learning occ implementation for a synthetic sklearn data

# importing all the necessary libraries
import numpy as np
import pandas as pd
import json
import csv
from datetime import datetime
from time import process_time

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
          
          curr_model[UN][wday]["Interval"] = interval
          curr_model[UN][wday]["IntervalCounter"][interval] += 1
          # curr_model[UN]["EndDate"].append(logdata["EndDate"])
          
          # Sources
          if logdata["SourceAddress"] not in curr_model[UN][wday]["SourceAddress"]:
               curr_model[UN][wday]["SourceAddress"][logdata["SourceAddress"]] = 1

          else:
               curr_model[UN][wday]["SourceAddress"][logdata["SourceAddress"]] += 1
          
          # Destinations
          if logdata["DestinationHost"] not in curr_model[UN][wday]["DestinationHost"]:
               curr_model[UN][wday]["DestinationHost"][logdata["DestinationHost"]] = 1

          else:
               curr_model[UN][wday]["DestinationHost"][logdata["DestinationHost"]] += 1
    
     else:
          curr_model[UN] = {}
          
          # [0,0,0,0,0,0,0,0]
          # For each wday we have the current interval, Int counter, SA and DH names
          curr_model[UN]['WD'] = {"IntervalCounter": [0,0,0,0,0,0,0,0], \
                                  "SourceAddress": {}, "DestinationHost": {}}
          curr_model[UN]['Sat'] = {"IntervalCounter": [0,0,0,0,0,0,0,0], \
                                  "SourceAddress": {}, "DestinationHost": {}}
          curr_model[UN]['Sun'] = {"IntervalCounter": [0,0,0,0,0,0,0,0], \
                                  "SourceAddress": {}, "DestinationHost": {}}
          ######################################

          # curr_model[UN]["UserLabel"] = len(curr_model)  # As soon as we specify a dict key (UN) its size increases by 1
          # curr_model[UN]["StartDate"] = list([logdata["StartDate"]])  # list of string timestamps
          curr_model[UN][wday]["Interval"] = interval
          curr_model[UN][wday]["IntervalCounter"][interval] += 1
          curr_model[UN][wday]["SourceAddress"] = {logdata["SourceAddress"]: 1} # 
          curr_model[UN][wday]["DestinationHost"] = {logdata["DestinationHost"]: 1} # list of 1D arrays

     return curr_model  

# Write a json file for the messages read from server i.e. a list of dictionaries 
def writeJson(filePathName, numLogs, readLines, names):

     with open(filePathName, 'r') as readFile:
          linecounter = 0

          for i in range(readLines+1):
               next(readFile) # skipping header line

          reader = csv.DictReader(readFile, fieldnames = names) # default fieldNames are the first row values
          dictList = []
          for row in reader:
               linecounter += 1
               if linecounter <= numLogs:
                    dictList.append(row)

               else:
                    break

          with open("logData.json", 'w') as f:
               json.dump(dictList, f)

          readFile.close()
          f.close()

def writeToModel(model, destination, filePathName, curr_date, numLogs, interval = 3):

     with open(filePathName, 'r') as file:
          reader = json.load(file)

     file.close()
     # reader = csv.reader(file, delimiter = ",")
     # counter and flag for the end of file
     linecounter = 0
     flag = 0
     # wday = None
     # destination = []
     for row in reader:
          record = {}
          linecounter += 1

          if linecounter <= numLogs:
               try:
                    logonType = int(row["DeviceCustomNumber1"])  # FirstLogin Device Custom Number1
                    # startDate = datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S")
                    Name = row["Name"]
                    startDate = row["StartDate"]  # timestamp in UTC
                    Date = datetime.utcfromtimestamp(int(startDate)/1000)

                    # Record userdata only for the present day and not the backdated data
                    if "logged on" in Name and logonType in [2,3,7,9,10] and Date.day == curr_date.day:

                         # Record the data
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

                         sourceAddress = row["SourceAddress"]
                         # sourceAddressList = convert2list(sourceAddress) # list of IPs
                         if len(sourceAddress) == 0:
                              source = 'NIL'
                         
                         else:
                              # source = IP2dec(sourceAddress)
                              source = sourceAddress

                         userName = row["DestinationUserName"]

                         destinationHostName = row["DestinationHostName"]  # Just one name
                         # destinationHostNameList =  convert2list(destinationHostName)

                         if destinationHostName in destination:
                              destinationLabel = destination.index(destinationHostName)

                         else:
                              destination.append(destinationHostName)  # it is a method with no return and appends the list with each
                         # element of the added list
                              destinationLabel = len(destination)-1
                         
                         if linecounter % numLogs == 0:
                              print(linecounter/ numLogs)

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
               break

     if linecounter < numLogs:
          flag = 1

     return model, destination, linecounter, flag

##################################################

# str = ["abc", "bcd", "839", "abc", "839", "aac"]
# tic = process_time()
# print(list(dict.fromkeys(str)))
# # print(np.unique(str))
# toc = process_time()