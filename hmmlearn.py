import constants
import sys
import datautil
import model
import numpy as np

# t1 = time.time()
train_file = sys.argv[1]
dev_file = sys.argv[2] if len(sys.argv) == 3 else None
# bestLaplace = 0.002

bestLaplace = 0.02

dataUtil = datautil.DataUtil(train_file=train_file, dev_file=dev_file)

# train_word_count, train_tag_count, train_word_tag_count, train_prevtag_tag_count = dataUtil.readTrainingData()

# hmmModel = model.Model(laplaceParam1=1, laplaceParam2=bestLaplace)
# hmmModel.calculateProbabilities(
#     train_word_count, train_tag_count, train_word_tag_count, train_prevtag_tag_count)
# hmmModel.dumpModel()
# if dev_file:
#     lines = dataUtil.readDevData()
#     predictedTags = hmmModel.predictPOSTags(constants.DEV, lines)
#     acc = hmmModel.checkAccuracy(lines, predictedTags)


# maxAcc = 0
# bestLaplace1 = None
# bestLaplace2 = None
# for laplaceParam1 in np.arange(1,5,0.5):
#     for laplaceParam2 in np.arange(0.02,1.01,0.2):
#         print(laplaceParam1,laplaceParam2)
#         hmmModel = model.Model(laplaceParam1=laplaceParam1,laplaceParam2=laplaceParam2)
#         hmmModel.calculateProbabilities(
#             train_word_count, train_tag_count, train_word_tag_count, train_prevtag_tag_count)
#         # hmmModel.dumpModel()
#         predictedTags = hmmModel.predictPOSTags(constants.DEV, lines)
#         acc = hmmModel.checkAccuracy(lines, predictedTags)
#         if acc>maxAcc:
#             maxAcc = acc
#             bestLaplace1 = laplaceParam1
#             bestLaplace2 = laplaceParam2
#             print("Best accuracy ",maxAcc,bestLaplace1,bestLaplace2)
# print("Best accuracy ",maxAcc, bestLaplace1, bestLaplace2)


train_word_count,train_tag_count,train_word_tag_count,train_prevtag_tag_count = dataUtil.readTrainAndDevData()
# t2 = time.time()

hmmModel = model.Model(laplaceParam2 = bestLaplace)
hmmModel.calculateProbabilities(train_word_count,train_tag_count,train_word_tag_count,train_prevtag_tag_count)
# t3 = time.time()

hmmModel.dumpModel()
# t4 = time.time()

# print("Read data ",t2-t1)
# print("Calc prob ",t3-t2)
# print("Dump model ",t4-t3)
# print("Training time ",t4-t1)