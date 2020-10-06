from Recognition import *
from GenerateData import *

CSV_FILE_NAME = 'DATA.csv'

# Generate Data CSV file for training neural network
print("------------------------------------------------------------------------------")
print("INITIALISING DATA")
print("------------------------------------------------------------------------------")
if os.path.exists(CSV_FILE_NAME):
    os.remove(CSV_FILE_NAME)
GetTrainingData(CSV_FILE_NAME)

print("------------------------------------------------------------------------------")
print("TRAINING MODEL")
print("------------------------------------------------------------------------------")
# Train and test Neural Network
mlp, X_train, y_train, X_val, y_val, X_test, y_test = Train(CSV_FILE_NAME, 'class')

# Printing out result
Print_Result(mlp, X_train, y_train, X_val, y_val, X_test, y_test)

