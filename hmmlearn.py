import constants
import sys
import datautil
import model
import time

# t1 = time.time()
train_file = sys.argv[1]
dev_file = sys.argv[2] if len(sys.argv) == 3 else None
# bestLaplace = 0.002

bestLaplace1 = 0.9
bestLaplace2 = 0.02
bestLaplace3 = 0.16

dataUtil = datautil.DataUtil(train_file=train_file, dev_file=dev_file)

# train_word_count, train_tag_count, train_word_tag_count, train_prevtag_tag_count = dataUtil.readTrainingData()

# hmmModel = model.Model(laplaceParam1=bestLaplace1, laplaceParam2=bestLaplace2, laplaceParam3=bestLaplace3)
# hmmModel.calculateProbabilities(
#     train_word_count, train_tag_count, train_word_tag_count, train_prevtag_tag_count)
# hmmModel.dumpModel()
# if dev_file:
#     lines = dataUtil.readDevData()
#     predictedTags = hmmModel.predictPOSTags(constants.DEV, lines)
#     acc = hmmModel.checkAccuracy(lines, predictedTags)


# maxAcc = 0
# bestLaplace1 = 0.9
# bestLaplace2 = 0.02
# bestLaplace3 = None
# for laplaceParam3 in np.arange(0.01,0.6,0.05):
#     hmmModel = model.Model(laplaceParam1=bestLaplace1,laplaceParam2=bestLaplace2, laplaceParam3=laplaceParam3)
#     hmmModel.calculateProbabilities(
#         train_word_count, train_tag_count, train_word_tag_count, train_prevtag_tag_count)
#     # hmmModel.dumpModel()
#     predictedTags = hmmModel.predictPOSTags(constants.DEV, lines)
#     acc = hmmModel.checkAccuracy(lines, predictedTags)
#     if acc>maxAcc:
#         maxAcc = acc
#         bestLaplace3 = laplaceParam3
#         print("Best accuracy ",maxAcc,bestLaplace3)
# print("Best accuracy ",maxAcc,  bestLaplace3)


train_word_count,train_tag_count,train_word_tag_count,train_prevtag_tag_count = dataUtil.readTrainAndDevData()
# t2 = time.time()

hmmModel = model.Model(laplaceParam1=bestLaplace1, laplaceParam2=bestLaplace2, laplaceParam3=bestLaplace3)
hmmModel.calculateProbabilities(train_word_count,train_tag_count,train_word_tag_count,train_prevtag_tag_count)
# t3 = time.time()

hmmModel.dumpModel()
# t4 = time.time()

# print("Read data ",t2-t1)
# print("Calc prob ",t3-t2)
# print("Dump model ",t4-t3)
# print("Training time ",t4-t1)