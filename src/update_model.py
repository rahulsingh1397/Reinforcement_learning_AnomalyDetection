# Sample code to check implementation
import json
from feedback_update_code import*

curr_date = datetime.strptime("2023-07-04", "%Y-%m-%d")

############################ LOAD JSON FILES #######################################
# Load the json file
state = input("Is the feedback & update model running for the first time? [Y/N]: ")

if state == 'Y':
    jsonName = "../models/TrainDataWeek_1.json"
else:
    jsonName = "../models/saveTrainDataUpdated.json"

with open(jsonName, 'r') as f:
    model_1 = json.load(f)

# Load the json file
jsonName = "../models/saveTestData.json"
with open(jsonName, 'r') as f:
    model_current = json.load(f)

# Load anomalous users keys
jsonName = "../outputs/AnomalousUsers.json"
with open(jsonName, 'r') as f:
    time_anomaly = json.load(f)

# Load anomalous source addresses
jsonName = "../outputs/AnomalousSource.json"
with open(jsonName, 'r') as f:
    source_anomaly = json.load(f)

# Load anomalous destination hosts
jsonName = "../outputs/AnomalousDestination.json"
with open(jsonName, 'r') as f:
    dest_anomaly = json.load(f)

# Load anomaly threshold for all the users
jsonName = "../outputs/AnomalyThreshold.json"
with open(jsonName, 'r') as f:
    threshold_dict = json.load(f)
#################################### LOAD JSON FILES ENDS ###################################


########## Call the feedback and corresponding train model update code #####################
model_new, threshold_dict = trainModelUpdate(curr_date, model_1, model_current, time_anomaly, source_anomaly, dest_anomaly, threshold_dict)

# ###########  Writing JSONs  ##############
with open("../models/saveTrainDataUpdated2.json", 'w') as f:
    json.dump(model_new, f)
f.close()

jsonName = "../outputs/AnomalyThreshold2.json"
with open(jsonName, 'w') as f:
    json.dump(threshold_dict, f)
f.close()

print("Model and anomaly thresholds have been updated")
################################# Writing json files ends ##################################