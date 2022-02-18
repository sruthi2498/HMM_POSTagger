
from collections import Counter
import constants
import json
import math


class Model:
    def __init__(self, mode=constants.TRAIN, laplaceParam1 = 0.05, laplaceParam2 = 0.05):
        self.model_file = "hmmmodel.txt"
        self.tags = []
        self.vocab = []
        self.emission_probs = {}
        self.transition_probs = {}
        self.laplace_param1 = laplaceParam1
        self.laplace_param2 = laplaceParam2
        self.tag_vocab_count = None
        if mode == constants.TEST:
            self.__fetchModel()

    def __laplace(self, c1, c2, V):
        return (c1+self.laplace_param1)/(c2 + (self.laplace_param2*V))

    def __extractTagVocabCount(self, word_tag_count):
        tag_vocab = {tag: set() for tag in self.tags if tag !=
                     constants.EOL_TAG and tag != constants.BOL_TAG}
        for word, tag in word_tag_count:
            tag_vocab[tag].add(word)
        return {tag: len(words) for tag, words in tag_vocab.items()}


    def extractOpenClassTags(self, word_tag_count):
        tag_vocab_count = self.__extractTagVocabCount(word_tag_count)
        total = len(self.vocab)
        perc = 0.04
        threshold = perc * total
        self.open_class_tags = {}
        while perc and not self.open_class_tags:
            # print("threshold = ", threshold)
            self.__extractHighCountTags(tag_vocab_count, threshold)
            perc -= 0.01
            threshold = perc * total
        self.tag_vocab_count = tag_vocab_count

    def __extractHighCountTags(self, tag_vocab_count, threshold):
        self.open_class_tags = {}
        for tag, count in tag_vocab_count.items():
            # print(tag,count)
            if tag != constants.BOL_TAG and tag != constants.EOL_TAG and count >= threshold:
                self.open_class_tags[tag] = count

    def calculateProbabilities(self, word_count, tag_count, word_tag_count, prevtag_tag_count):
        self.tags = list(tag_count.keys())
        self.vocab = list(word_count.keys())
        totalTags = len(self.tags)

        for tag in self.tags:
            self.emission_probs[tag] = {w: 0 for w in self.vocab}
            self.transition_probs[tag] = {t2: 0 for t2 in self.tags}

        # print("Calculating probabilities")
        # for word_tag, count in word_tag_count.items():
        #     word, tag = word_tag
        #     # P(word|tag) = c(word,tag)/c(tag)
        #     #emission_probs[tag][word] = P(word|tag)
        #     self.emission_probs[tag][word] = self.__laplace(
        #         count, tag_count[tag], totalTags)

        for tag in self.tags:
            for word in self.vocab:
                if word!=constants.UNKNOWN_WORD and (word,tag)  in word_tag_count:
                    # self.emission_probs[tag][word] = self.__laplace(word_tag_count[(word,tag)], tag_count[tag], totalTags)
                    self.emission_probs[tag][word] = word_tag_count[(word,tag)] /tag_count[tag]

        # for tags, count in prevtag_tag_count.items():
        #     prev_tag, tag = tags
        #     # P(tag|prev_tag) = c(prev_tag,tag)/c(prev_tag)
        #     #transition_probs[prev_tag][tag] = P(tag|prev_tag)
        #     self.transition_probs[prev_tag][tag] = self.__laplace(
        #         count, tag_count[prev_tag], totalTags)

        for prev_tag in self.tags:
            for tag in self.tags:
                count = 0 if  (prev_tag,tag) not in prevtag_tag_count else prevtag_tag_count[(prev_tag,tag)]
                self.transition_probs[prev_tag][tag] = self.__laplace(count, tag_count[prev_tag], totalTags)
                
                    

        self.extractOpenClassTags(word_tag_count)

    def dumpModel(self):
        #print("Dumping model to ", self.model_file)
        model = {
            constants.MODEL_TAGS: self.tags,
            constants.MODEL_VOCAB: self.vocab,
            constants.MODEL_EMISSION_PROBS: self.emission_probs,
            constants.MODEL_TRANSITION_PROBS: self.transition_probs,
            constants.MODEL_OPEN_CLASS_TAGS: self.open_class_tags
        }
        with open(self.model_file, 'w') as f:
            json.dump(model, f)

    def __fetchModel(self):
        with open(self.model_file) as f:
            #print("Reading model from ", self.model_file)
            model = json.load(f)
            self.tags = model[constants.MODEL_TAGS]
            self.vocab = model[constants.MODEL_VOCAB]
            self.emission_probs = model[constants.MODEL_EMISSION_PROBS]
            self.transition_probs = model[constants.MODEL_TRANSITION_PROBS]
            self.open_class_tags = model[constants.MODEL_OPEN_CLASS_TAGS]
            #print("Tags : ", len(self.tags), "OpenClass Tags : ", len(self.open_class_tags), "Vocab : ", len(self.vocab))

    def __getWord(self, word):
        if word in self.vocab:
            return word
        if word.lower() in self.vocab:
            return word.lower()
        if word.upper() in self.vocab:
            return word.upper()
        return constants.UNKNOWN_WORD

    def __performViterbi(self, line):
        T = len(line)
        probabiities = {
            tag: [-math.inf for _ in range(T+1)] for tag in self.tags}
        backpointer = {tag: [None for _ in range(T+1)] for tag in self.tags}
        word = self.__getWord(line[0])

        if word == constants.UNKNOWN_WORD:
            for tag in self.open_class_tags:
                probabiities[tag][0] = math.log(
                        self.transition_probs[constants.BOL_TAG][tag])

                backpointer[tag][0] = constants.BOL_TAG
        else:
            for tag in self.tags:
                if self.emission_probs[tag][word] and self.transition_probs[constants.BOL_TAG][tag]  :
                    probabiities[tag][0] = math.log(
                        self.transition_probs[constants.BOL_TAG][tag]) + math.log(self.emission_probs[tag][word])

                backpointer[tag][0] = constants.BOL_TAG

        for t in range(1, T):
            word = self.__getWord(line[t])

            if word == constants.UNKNOWN_WORD:
                # print("Unknown word")
                for tag in self.open_class_tags:
                    maxProb = -math.inf
                    maxPrevState = None
                    for prev_tag in self.tags:
                        if self.transition_probs[prev_tag][tag] and probabiities[prev_tag][t-1] != -math.inf :

                            prob = probabiities[prev_tag][t-1] + \
                                math.log(self.transition_probs[prev_tag][tag] ) 
                            
                            if prob > maxProb :
                                maxProb = prob
                                maxPrevState = prev_tag

                    probabiities[tag][t] = maxProb - math.log(self.open_class_tags[tag]/len(self.vocab))
                    backpointer[tag][t] = maxPrevState
            else:
                for tag in self.tags:
                    if tag != constants.BOL_TAG and tag != constants.EOL_TAG:
                        maxProb = -math.inf
                        maxPrevState = None
                        for prev_tag in self.tags:
                            if  self.emission_probs[tag][word] and \
                                    self.transition_probs[prev_tag][tag] and \
                                    probabiities[prev_tag][t-1] != -math.inf :

                                prob = probabiities[prev_tag][t-1] + math.log(
                                    self.transition_probs[prev_tag][tag]) + math.log(self.emission_probs[tag][word])

                                if prob > maxProb:
                                    maxProb = prob
                                    maxPrevState = prev_tag

                        probabiities[tag][t] = maxProb
                        backpointer[tag][t] = maxPrevState


        mostProbableLastState = None
        mostProbableLastStateProb = -math.inf
        for tag in self.tags:
            prob = probabiities[tag][T-1]+ math.log(
                    self.transition_probs[prev_tag][constants.EOL_TAG]) 
            if prob > mostProbableLastStateProb:
                mostProbableLastStateProb = probabiities[tag][T-1]
                mostProbableLastState = tag

        return self.__backtrack(T, backpointer, mostProbableLastState)

    def __backtrack(self, T, backpointer, mostProbableLastState):
        # print("mostProbableLastState = ",mostProbableLastState)
        result = []
        tag = mostProbableLastState
        i = T-1
        while i >= 0:
            if not tag:
                break
            result.append(tag)
            tag = backpointer[tag][i]
            i -= 1
        result.reverse()
        return result

    def predictPOSTags(self, mode, lines):
        if mode == constants.DEV:
            wordsOnlyLine = []
            for line in lines:
                wordsOnlyLine.append([word for word, _ in line])

        elif mode == constants.TEST:
            wordsOnlyLine = lines

        result = []
        for i, line in enumerate(wordsOnlyLine):
            tags = self.__performViterbi(line)
            # print(len(line), len(tags),tags)
            # print(lines[i])
            result.append(tags)

        return result

    def checkAccuracy(self, lines, predictedTags):
        correct = 0
        total = 0
        i = 0
        wrong_preds = {}
        wrong_preds_unknownwords = {}
        for line, predicted in zip(lines, predictedTags):
            if len(line) != len(predicted):
                print("Something wrong", i)
                print(line)
                print(predicted)
                break
            for (word, tag), predictedTag in zip(line, predicted):
                total += 1
                if tag == predictedTag:
                    correct += 1
                else:
                    if (predictedTag,tag) in wrong_preds:
                        wrong_preds[(predictedTag,tag)]+=1
                    else:
                        wrong_preds[(predictedTag,tag)]=1
                     
                    if word not in self.vocab:
                        if (predictedTag,tag) in wrong_preds_unknownwords:
                            wrong_preds_unknownwords[(predictedTag,tag)]+=1
                        else:
                            wrong_preds_unknownwords[(predictedTag,tag)]=1


            i += 1
        
        wrong_preds = sorted(wrong_preds.items(), key=lambda x : x[1],reverse = True)
        if self.tag_vocab_count:
            for k,v in wrong_preds[:5]:
                if v>1 and k in wrong_preds_unknownwords:
                    print(k,v,wrong_preds_unknownwords[k])
                    print("\t",k[0],self.tag_vocab_count[k[0]],k[1],self.tag_vocab_count[k[1]])
                    print("\t",k[1] ,"in self.open_class_tags : ",k[1] in self.open_class_tags)

        print("Accuracy : ", correct/total)
        print(self.open_class_tags) 
        return correct/total