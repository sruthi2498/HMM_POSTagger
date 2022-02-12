import constants
import json
import math

class Model:
    def __init__(self, mode = constants.TRAIN):
        self.model_file = "hmmmodel.txt"
        self.tags = []
        self.vocab = []
        self.emission_probs= {}
        self.transition_probs = {}
        if mode == constants.TEST:
            self.__fetchModel()

    def __laplace(self,c1, c2,V):
        lamb = 0.001
        return (c1+lamb)/(c2 + (lamb*V))

    def calculateProbabilities(self,word_count,tag_count,word_tag_count,prevtag_tag_count):
        self.tags = list(tag_count.keys())
        self.vocab = list(word_count.keys())
        totalTags = len(self.tags)
        print("Tags : ",len(self.tags), "Vocab : ",len(self.vocab))

        for tag in self.tags:
            self.emission_probs[tag] = {w : 0 for w in self.vocab}
            self.transition_probs[tag] = {t2 : 0 for t2 in self.tags}

        print("Calculating probabilities")
        for word_tag,count in word_tag_count.items():
            word,tag = word_tag
            #P(word|tag) = c(word,tag)/c(tag)
            #emission_probs[tag][word] = P(word|tag)
            self.emission_probs[tag][word] = self.__laplace(count,tag_count[tag],totalTags)
            
        for tag in self.tags:
            for word in self.vocab:
                if not self.emission_probs[tag][word]:
                    self.emission_probs[tag][word] = self.__laplace(0,tag_count[tag],totalTags)

        # for tag in self.tags:
        #     for word in self.vocab:
        #         if self.emission_probs[tag][word]<0:
        #             print(self.emission_probs[tag][word])

        for tags,count in prevtag_tag_count.items():
            prev_tag,tag = tags
            #P(tag|prev_tag) = c(prev_tag,tag)/c(prev_tag)
            #transition_probs[prev_tag][tag] = P(tag|prev_tag)
            self.transition_probs[prev_tag][tag] =  self.__laplace(count,tag_count[prev_tag],totalTags)


        for prev_tag in self.tags:
            for tag in self.tags:
                if not self.transition_probs[prev_tag][tag]:
                    self.transition_probs[prev_tag][tag] = self.__laplace(0,tag_count[prev_tag],totalTags)

        # for tag in self.tags:
        #     print(self.transition_probs[constants.BOL_TAG][tag])

    def dumpModel(self):
        print("Dumping model to ",self.model_file)
        model = {
            constants.MODEL_TAGS : self.tags,
            constants.MODEL_VOCAB : self.vocab,
            constants.MODEL_EMISSION_PROBS : self.emission_probs,
            constants.MODEL_TRANSITION_PROBS : self.transition_probs
        }
        with open(self.model_file, 'w') as f:
            json.dump(model, f)

    def __fetchModel(self):
        with open(self.model_file) as f:
            print("Reading model from ",self.model_file)
            model = json.load(f)
            self.tags = model[constants.MODEL_TAGS]
            self.vocab = model[constants.MODEL_VOCAB]
            self.emission_probs= model[constants.MODEL_EMISSION_PROBS]
            self.transition_probs = model[constants.MODEL_TRANSITION_PROBS]
            print("Tags : ",len(self.tags), "Vocab : ",len(self.vocab))

    def __getWord(self,word):
        if word in self.vocab:
            return word
        return constants.UNKNOWN_WORD
            
    def __performViterbi(self, line):
        T = len(line) 
        probabiities = {tag:[-math.inf for _ in range(T)] for tag in self.tags}
        backpointer = {tag:[None for _ in range(T)] for tag in self.tags}
        word = self.__getWord(line[0])

        for tag in self.tags:
            if self.transition_probs[constants.BOL_TAG][tag] and self.emission_probs[tag][word]:
                probabiities[tag][0] = math.log( self.transition_probs[constants.BOL_TAG][tag] ) + math.log( self.emission_probs[tag][word] )
                
            backpointer[tag][0] = constants.BOL_TAG 


        for t in range(1,T):
            word = self.__getWord(line[t])
            for tag in self.tags:
                if tag!=constants.BOL_TAG and tag!=constants.EOL_TAG:
                    maxProb = -math.inf
                    maxPrevState = None

                    for prev_tag in self.tags:
                        # if t==T-1 and probabiities[prev_tag][t-1]!=-math.inf:
                        #     print(prev_tag in self.transition_probs and tag in self.transition_probs[prev_tag] , 
                        #     tag in self.emission_probs and word in self.emission_probs[tag])
                        if probabiities[prev_tag][t-1]!=-math.inf and\
                            prev_tag in self.transition_probs and tag in self.transition_probs[prev_tag] and \
                            tag in self.emission_probs and word in self.emission_probs[tag] :
                            
                            prob = probabiities[prev_tag][t-1] + math.log( self.transition_probs[prev_tag][tag] ) + math.log(self.emission_probs[tag][word])

                            if prob>maxProb:
                                maxProb = prob
                                maxPrevState = prev_tag
                        
                    probabiities[tag][t] = maxProb
                    backpointer[tag][t] = maxPrevState


        mostProbableLastState = None
        mostProbableLastStateProb = -math.inf
        for tag in self.tags:
            if probabiities[tag][T-1] > mostProbableLastStateProb:
                mostProbableLastStateProb = probabiities[tag][T-1]
                mostProbableLastState = tag 
        
        return self.__backtrack(T, backpointer,mostProbableLastState)
        

    def __backtrack(self,T, backpointer, mostProbableLastState):
        # print("mostProbableLastState = ",mostProbableLastState)
        result = []
        tag = mostProbableLastState
        i = T-1
        while i>=0:
            if not tag:
                break
            result.append(tag)
            tag = backpointer[tag][i]
            i-=1
        result.reverse()
        return result


    def predictPOSTags(self, mode, lines):
        if mode==constants.DEV:
            wordsOnlyLine = []
            for line in lines:
                wordsOnlyLine.append([ word for word,_ in line ])
            
        elif mode==constants.TEST:
            wordsOnlyLine = lines 
        
        result = []
        for i,line in enumerate(wordsOnlyLine):
            tags= self.__performViterbi(line)
            # print(len(line), len(tags),tags)
            # print(lines[i][:21])
            result.append(tags)

        return result

    def checkAccuracy(self, lines, predictedTags):
        correct = 0
        total = 0
        i = 0
        for line, predicted in zip(lines,predictedTags):
            if len(line)!=len(predicted):
                print("Something wrong",i)
                break
            for (_,tag),predictedTag in zip(line,predicted):
                total+=1
                if tag==predictedTag:
                    correct+=1
            i+=1
        print("Accuracy : ",correct/total)