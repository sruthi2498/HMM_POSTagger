import sys
import datautil
import model
import constants
import time

# t1 = time.time()
test_file = sys.argv[1]
dataUtil = datautil.DataUtil(test_file=test_file)
lines = dataUtil.readTestData()
# t2 = time.time()

hmmModel = model.Model(constants.TEST)
# t3 = time.time()
predictedTags = hmmModel.predictPOSTags(constants.TEST, lines)
# t4 = time.time()

dataUtil.dumpOutput(lines, predictedTags)
# t5 = time.time()

# print("Read data ",t2-t1)
# print("Fetch model ",t3-t2)
# print("Predict ",t4-t3)
# print("Dump output ",t5-t4)
# print("Testing time ",t5-t1)