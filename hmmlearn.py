import constants
import sys
import datautil
import model
import numpy as np

train_file = sys.argv[1]
dev_file = sys.argv[2] if len(sys.argv) == 3 else None

dataUtil = datautil.DataUtil(train_file=train_file, dev_file=dev_file)

train_word_count, train_tag_count, train_word_tag_count, train_prevtag_tag_count = dataUtil.readTrainingData()

bestLaplace = 0.002
hmmModel = model.Model(laplaceParam=bestLaplace)
hmmModel.calculateProbabilities(
    train_word_count, train_tag_count, train_word_tag_count, train_prevtag_tag_count)
hmmModel.dumpModel()
if dev_file:
    lines = dataUtil.readDevData()
    predictedTags = hmmModel.predictPOSTags(constants.DEV, lines)
    acc = hmmModel.checkAccuracy(lines, predictedTags)

# maxAcc = 0
# bestLaplace = None
# for laplaceParam in np.arange(5e-4, 5e-3, 5e-4):
#     print(laplaceParam)
#     hmmModel = model.Model(laplaceParam = laplaceParam)
#     hmmModel.calculateProbabilities(
#         train_word_count, train_tag_count, train_word_tag_count, train_prevtag_tag_count)
#     # hmmModel.dumpModel()
#     predictedTags = hmmModel.predictPOSTags(constants.DEV, lines)
#     acc = hmmModel.checkAccuracy(lines, predictedTags)
#     if acc>maxAcc:
#         maxAcc = acc
#         bestLaplace = laplaceParam
#         print("Best accuracy ",maxAcc, " laplaceParam = ",bestLaplace)
# print("Best accuracy ",maxAcc, " laplaceParam = ",bestLaplace)


train_word_count,train_tag_count,train_word_tag_count,train_prevtag_tag_count = dataUtil.readTrainAndDevData()
hmmModel = model.Model(laplaceParam = bestLaplace)
hmmModel.calculateProbabilities(train_word_count,train_tag_count,train_word_tag_count,train_prevtag_tag_count)
hmmModel.dumpModel()
