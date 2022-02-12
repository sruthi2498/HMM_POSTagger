import constants


class DataUtil:
    def __init__(self, train_file=None, dev_file=None, test_file=None):
        self.train_file = train_file
        self.output_file = "hmmoutput.txt"
        self.dev_file = dev_file
        self.test_file = test_file

    def __getFileLines(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [l.replace("\n", "").replace("\r", "").strip()
                     for l in lines]
            print("Lines : ", len(lines))
            return lines

    def readTrainingData(self):
        if self.train_file:
            print("Reading from ", self.train_file)
            return self.__readTaggedData(constants.TRAIN)

    def readDevData(self):
        if self.dev_file:
            print("Reading from ", self.dev_file)
            return self.__readTaggedData(constants.DEV)

    def readTrainAndDevData(self):
        if self.dev_file:
            print("Reading from ", self.train_file, " and ", self.dev_file)
            return self.__readTaggedData(constants.TRAIN_DEV)
        print("Reading from ", self.train_file)
        return self.__readTaggedData(constants.TRAIN)

    def readTestData(self):
        if self.test_file:
            print("Reading from ", self.test_file)
            lines = [l.split(" ") for l in self.__getFileLines(self.test_file)]
            return lines

    def __readTaggedData(self, mode):
        if mode == constants.TRAIN:
            lines = self.__getFileLines(self.train_file)
        elif mode == constants.DEV:
            lines = self.__getFileLines(self.dev_file)
            lines = [l.split(" ") for l in lines]
            for i, line in enumerate(lines):
                for j, word_tag in enumerate(line):
                    word_tag = word_tag.split("/")
                    word = "".join(word_tag[:-1])
                    tag = word_tag[-1].upper()
                    lines[i][j] = (word, tag)
            return lines
        elif mode == constants.TRAIN_DEV:
            lines = self.__getFileLines(
                self.dev_file) + self.__getFileLines(self.train_file)

        word_count = {constants.UNKNOWN_WORD: 0}
        # Count tag occurences
        tag_count = {constants.EOL_TAG: 0, constants.BOL_TAG: 0}
        for i, line in enumerate(lines):
            line = line.split(" ")
            lines[i] = line
            for j, word_tag in enumerate(line):
                word_tag = word_tag.split("/")
                word = "".join(word_tag[:-1])
                tag = word_tag[-1]
                lines[i][j] = (word, tag)
                if tag not in tag_count:
                    tag_count[tag] = 1
                else:
                    tag_count[tag] += 1

                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1
            # begin of line tag for each line
            tag_count[constants.BOL_TAG] += 1
            tag_count[constants.EOL_TAG] += 1  # end of line tag for each line

        # Count word,tag pair occurences
        word_tag_count = {}
        for line in lines:
            for word_tag in line:
                if word_tag in word_tag_count:
                    word_tag_count[word_tag] += 1
                else:
                    word_tag_count[word_tag] = 1

        # Count tag,tag pair occurences
        prevtag_tag_count = {}
        for line in lines:
            prev_tag = line[0][1]
            tag_pair = (constants.BOL_TAG, prev_tag)
            if tag_pair not in prevtag_tag_count:
                prevtag_tag_count[tag_pair] = 1
            else:
                prevtag_tag_count[tag_pair] += 1

            for _, tag in line[1:]:
                tag_pair = (prev_tag, tag)
                if tag_pair not in prevtag_tag_count:
                    prevtag_tag_count[tag_pair] = 1
                else:
                    prevtag_tag_count[tag_pair] += 1
                prev_tag = tag

            tag_pair = (tag, constants.EOL_TAG)
            if tag_pair not in prevtag_tag_count:
                prevtag_tag_count[tag_pair] = 1
            else:
                prevtag_tag_count[tag_pair] += 1

        return word_count, tag_count, word_tag_count, prevtag_tag_count

    def dumpOutput(self, lines, predictedTags):
        result = ""
        for line, tags in zip(lines, predictedTags):
            sentence = ""
            for word, tag in zip(line, tags):
                sentence += word
                sentence += "/"
                sentence += tag
                sentence += " "
            sentence += "\n"
            result += sentence
        with open(self.output_file, "w") as f:
            f.write(result)
