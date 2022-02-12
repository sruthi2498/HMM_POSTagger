import constants
import sys
import datautil
import model

train_file = sys.argv[1]
dev_file = sys.argv[2] if len(sys.argv)==3 else None
dataUtil = datautil.DataUtil(train_file=train_file,dev_file=dev_file)
train_word_count,train_tag_count,train_word_tag_count,train_prevtag_tag_count = dataUtil.readTrainingData()

hmmModel = model.Model()
hmmModel.calculateProbabilities(train_word_count,train_tag_count,train_word_tag_count,train_prevtag_tag_count)
hmmModel.dumpModel()

# if dev_file:
#     lines = dataUtil.readDevData()
#     predictedTags = hmmModel.predictPOSTags(constants.DEV,lines)
#     hmmModel.checkAccuracy(lines,predictedTags)

train_word_count,train_tag_count,train_word_tag_count,train_prevtag_tag_count = dataUtil.readTrainAndDevData()
hmmModel = model.Model()
hmmModel.calculateProbabilities(train_word_count,train_tag_count,train_word_tag_count,train_prevtag_tag_count)
hmmModel.dumpModel()

