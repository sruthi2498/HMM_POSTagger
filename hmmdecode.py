import sys
import datautil
import model
import constants

test_file = sys.argv[1]
dataUtil = datautil.DataUtil(test_file=test_file)
lines = dataUtil.readTestData()

hmmModel = model.Model(constants.TEST)
predictedTags = hmmModel.predictPOSTags(constants.TEST, lines)

dataUtil.dumpOutput(lines, predictedTags)
