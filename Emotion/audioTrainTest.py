import numpy
import cPickle
import sklearn.svm
import sklearn.decomposition
import sklearn.ensemble


def classifierWrapper(classifier, classifierType, testSample):
    R = classifier.predict(testSample.reshape(1, -1))[0]
    P = classifier.predict_proba(testSample.reshape(1, -1))[0]
    return [R, P]


def randSplitFeatures(features, partTrain):
    featuresTrain = []
    featuresTest = []
    for i, f in enumerate(features):
        [numOfSamples, numOfDims] = f.shape
        randperm = numpy.random.permutation(range(numOfSamples))
        nTrainSamples = int(round(partTrain * numOfSamples))
        featuresTrain.append(f[randperm[0:nTrainSamples]])
        featuresTest.append(f[randperm[nTrainSamples::]])
    return (featuresTrain, featuresTest)


def trainSVM(features, Cparam):
    [X, Y] = listOfFeatures2Matrix(features)
    svm = sklearn.svm.SVC(C=Cparam, kernel='linear', probability=True)
    svm.fit(X, Y)
    return svm


def trainRandomForest(features, n_estimators):
    [X, Y] = listOfFeatures2Matrix(features)
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators)
    rf.fit(X, Y)
    return rf


def trainGradientBoosting(features, n_estimators):
    [X, Y] = listOfFeatures2Matrix(features)
    rf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=n_estimators)
    rf.fit(X, Y)
    return rf


def trainExtraTrees(features, n_estimators):
    [X, Y] = listOfFeatures2Matrix(features)
    et = sklearn.ensemble.ExtraTreesClassifier(n_estimators=n_estimators)
    et.fit(X, Y)
    return et


def featureAndTrain(classifierType, modelName, computeBEAT=False, perTrain=0.9, arff_file=""):
    features, classNames, featureNames = readTrainDataFromARFF(arff_file)
    if classifierType == "svm":
        classifierParams = numpy.array([0.001, 0.01,  0.5, 1.0, 5.0, 10.0])
        bestParam = 0.0011
    elif classifierType == "randomforest":
        classifierParams = numpy.array([10, 25, 50, 100,200,500])
        bestParam = 1400
    elif classifierType == "gradientboosting":
        classifierParams = numpy.array([10, 25, 50, 100,200,500])
        bestParam = 350
    elif classifierType == "extratrees":
        classifierParams = numpy.array([10, 25, 50, 100,200,500])
        bestParam = 520

    bestParam = evaluateClassifier(features, classNames, 100, classifierType, classifierParams, 0, perTrain)
    print "Selected params: {0:.5f}".format(bestParam)
    [featuresNorm, MEAN, STD] = normalizeFeatures(features)  # normalize features
    MEAN = MEAN.tolist()
    STD = STD.tolist()
    featuresNew = featuresNorm

    # STEP C: Save the classifier to file
    if classifierType == "svm":
        Classifier = trainSVM(featuresNew, bestParam)
        with open(modelName, 'wb') as fid:  # save to file
            cPickle.dump(Classifier, fid)
        fo = open(modelName + "MEANS", "wb")
        cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(STD, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(classNames, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(computeBEAT, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        fo.close()
    elif classifierType == "randomforest":
        Classifier = trainRandomForest(featuresNew, bestParam)
        with open(modelName, 'wb') as fid:  # save to file
            cPickle.dump(Classifier, fid)
        fo = open(modelName + "MEANS", "wb")
        cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(STD, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(classNames, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(computeBEAT, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        fo.close()
    elif classifierType == "gradientboosting":
        Classifier = trainGradientBoosting(featuresNew, bestParam)
        with open(modelName, 'wb') as fid:  # save to file
            cPickle.dump(Classifier, fid)
        fo = open(modelName + "MEANS", "wb")
        cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(STD, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(classNames, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(computeBEAT, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        fo.close()
    elif classifierType == "extratrees":
        Classifier = trainExtraTrees(featuresNew, bestParam)
        with open(modelName, 'wb') as fid:  # save to file
            cPickle.dump(Classifier, fid)
        fo = open(modelName + "MEANS", "wb")
        cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(STD, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(classNames, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(computeBEAT, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        fo.close()

def loadSVModel(SVMmodelName):
    try:
        fo = open(SVMmodelName + "MEANS", "rb")
    except IOError:
        print "Load SVM Model: Didn't find file"
        return
    try:
        MEAN = cPickle.load(fo)
        STD = cPickle.load(fo)
        classNames = cPickle.load(fo)

    except:
        fo.close()
    fo.close()

    MEAN = numpy.array(MEAN)
    STD = numpy.array(STD)
    with open(SVMmodelName, 'rb') as fid:
        SVM = cPickle.load(fid)
    return (SVM, MEAN, STD, classNames)


def loadRandomForestModel(RFmodelName):
    try:
        fo = open(RFmodelName + "MEANS", "rb")
    except IOError:
        print "Load Random Forest Model: Didn't find file"
        return
    try:
        MEAN = cPickle.load(fo)
        STD = cPickle.load(fo)
        classNames = cPickle.load(fo)

    except:
        fo.close()
    fo.close()

    MEAN = numpy.array(MEAN)
    STD = numpy.array(STD)
    with open(RFmodelName, 'rb') as fid:
        RF = cPickle.load(fid)
    return (RF, MEAN, STD, classNames)


def loadGradientBoostingModel(GBModelName):
    try:
        fo = open(GBModelName + "MEANS", "rb")
    except IOError:
        print "Load Random Forest Model: Didn't find file"
        return
    try:
        MEAN = cPickle.load(fo)
        STD = cPickle.load(fo)
        classNames = cPickle.load(fo)

    except:
        fo.close()
    fo.close()

    MEAN = numpy.array(MEAN)
    STD = numpy.array(STD)

    with open(GBModelName, 'rb') as fid:
        GB = cPickle.load(fid)

    return (GB, MEAN, STD, classNames)


def loadExtraTreesModel(ETmodelName):
    try:
        fo = open(ETmodelName + "MEANS", "rb")
    except IOError:
        print "Load Random Forest Model: Didn't find file"
        return
    try:
        MEAN = cPickle.load(fo)
        STD = cPickle.load(fo)
        classNames = cPickle.load(fo)

    except:
        fo.close()
    fo.close()

    MEAN = numpy.array(MEAN)
    STD = numpy.array(STD)

    with open(ETmodelName, 'rb') as fid:
        GB = cPickle.load(fid)

    return (GB, MEAN, STD, classNames)


def evaluateClassifier(features, ClassNames, nExp, ClassifierName, Params, parameterMode, perTrain=0.90):
    (featuresNorm, MEAN, STD) = normalizeFeatures(features)
    nClasses = len(features)
    acAll = []
    F1All = []
    PrecisionClassesAll = []
    RecallClassesAll = []
    F1ClassesAll = []
    CMsAll = []

    nSamplesTotal = 0
    for f in features:
        nSamplesTotal += f.shape[0]
    for Ci, C in enumerate(Params):  # for each param value
        CM = numpy.zeros((nClasses, nClasses))
        for e in range(nExp):  # for each cross-validation iteration:
            print "Param = {0:.5f} - Classifier Evaluation Experiment {1:d} of {2:d}".format(C, e + 1, nExp)
            featuresTrain, featuresTest = randSplitFeatures(featuresNorm, perTrain)
            if ClassifierName == "svm":
                Classifier = trainSVM(featuresTrain, C)
            elif ClassifierName == "randomforest":
                Classifier = trainRandomForest(featuresTrain, C)
            elif ClassifierName == "gradientboosting":
                Classifier = trainGradientBoosting(featuresTrain, C)
            elif ClassifierName == "extratrees":
                Classifier = trainExtraTrees(featuresTrain, C)

            CMt = numpy.zeros((nClasses, nClasses))
            for c1 in range(nClasses):
                nTestSamples = len(featuresTest[c1])
                Results = numpy.zeros((nTestSamples, 1))
                for ss in range(nTestSamples):
                    [Results[ss], _] = classifierWrapper(Classifier, ClassifierName, featuresTest[c1][ss])
                for c2 in range(nClasses):
                    CMt[c1][c2] = float(len(numpy.nonzero(Results == c2)[0]))
            CM = CM + CMt
        CM = CM + 0.0000000010
        Rec = numpy.zeros((CM.shape[0],))
        Pre = numpy.zeros((CM.shape[0],))

        for ci in range(CM.shape[0]):
            Rec[ci] = CM[ci, ci] / numpy.sum(CM[ci, :])
            Pre[ci] = CM[ci, ci] / numpy.sum(CM[:, ci])
        PrecisionClassesAll.append(Pre)
        RecallClassesAll.append(Rec)
        F1 = 2 * Rec * Pre / (Rec + Pre)
        F1ClassesAll.append(F1)
        acAll.append(numpy.sum(numpy.diagonal(CM)) / numpy.sum(CM))

        CMsAll.append(CM)
        F1All.append(numpy.mean(F1))

    print ("\t\t"),
    for i, c in enumerate(ClassNames):
        if i == len(ClassNames) - 1:
            print "{0:s}\t\t".format(c),
        else:
            print "{0:s}\t\t\t".format(c),
    print ("OVERALL")
    print ("\tC"),
    for c in ClassNames:
        print "\tPRE\tREC\tF1",
    print "\t{0:s}\t{1:s}".format("ACC", "F1")
    bestAcInd = numpy.argmax(acAll)
    bestF1Ind = numpy.argmax(F1All)
    for i in range(len(PrecisionClassesAll)):
        print "\t{0:.3f}".format(Params[i]),
        for c in range(len(PrecisionClassesAll[i])):
            print "\t{0:.1f}\t{1:.1f}\t{2:.1f}".format(100.0 * PrecisionClassesAll[i][c],
                                                       100.0 * RecallClassesAll[i][c], 100.0 * F1ClassesAll[i][c]),
        print "\t{0:.1f}\t{1:.1f}".format(100.0 * acAll[i], 100.0 * F1All[i]),
        if i == bestF1Ind:
            print "\t best F1",
        if i == bestAcInd:
            print "\t best Acc",
        print

    if parameterMode == 0:  # keep parameters that maximize overall classification accuracy:
        print "Confusion Matrix:"
        printConfusionMatrix(CMsAll[bestAcInd], ClassNames)
        return Params[bestAcInd]
    elif parameterMode == 1:  # keep parameters that maximize overall F1 measure:
        print "Confusion Matrix:"
        printConfusionMatrix(CMsAll[bestF1Ind], ClassNames)
        return Params[bestF1Ind]


def printConfusionMatrix(CM, ClassNames):
    if CM.shape[0] != len(ClassNames):
        print "printConfusionMatrix: Wrong argument sizes\n"
        return

    for c in ClassNames:
        if len(c) > 4:
            c = c[0:3]
        print "\t{0:s}".format(c),
    print

    for i, c in enumerate(ClassNames):
        if len(c) > 4:
            c = c[0:3]
        print "{0:s}".format(c),
        for j in range(len(ClassNames)):
            print "\t{0:.1f}".format(100.0 * CM[i][j] / numpy.sum(CM)),
        print


def normalizeFeatures(features):
    X = numpy.array([])

    for count, f in enumerate(features):
        if f.shape[0] > 0:
            if count == 0:
                X = f
            else:
                X = numpy.vstack((X, f))
            count += 1

    MEAN = numpy.mean(X, axis=0)
    STD = numpy.std(X, axis=0)

    featuresNorm = []
    for f in features:
        ft = f.copy()
        for nSamples in range(f.shape[0]):
            ft[nSamples, :] = (ft[nSamples, :] - MEAN) / STD
        featuresNorm.append(ft)

    import math
    for feature in featuresNorm:
        for i in range(feature.shape[0]):
            for j in range(feature.shape[1]):
                if math.isnan(feature[i][j]):
                    feature[i][j] = 0

    return (featuresNorm, MEAN, STD)


def listOfFeatures2Matrix(features):
    X = numpy.array([])
    Y = numpy.array([])
    for i, f in enumerate(features):
        if i == 0:
            X = f
            Y = i * numpy.ones((len(f), 1))
        else:
            X = numpy.vstack((X, f))
            Y = numpy.append(Y, i * numpy.ones((len(f), 1)))
    return (X, Y)


def fileClassification(inputFile, modelName, modelType):
    if modelType == 'svm':
        [Classifier, MEAN, STD, classNames] = loadSVModel(modelName)
    elif modelType == 'randomforest':
        [Classifier, MEAN, STD, classNames] = loadRandomForestModel(modelName)
    elif modelType == 'gradientboosting':
        [Classifier, MEAN, STD, classNames] = loadGradientBoostingModel(modelName)
    elif modelType == 'extratrees':
        [Classifier, MEAN, STD, classNames] = loadExtraTreesModel(modelName)

    import math
    MidTermFeatures = readoneDataFromARFF(inputFile)
    curFV = (MidTermFeatures - MEAN) / STD
    for i in range(curFV.shape[0]):
        if math.isnan(curFV[i]):
            curFV[i] = 0

    [Result, P] = classifierWrapper(Classifier, modelType, curFV)  # classification
    return Result, P, classNames


def readTrainDataFromARFF(arff_filename):
    import re
    class_names = ("angry", "fear", "happy", "neutral", "sad", "surprise")
    f = open(arff_filename)
    line = f.readline()
    model_name = re.match("@relation (.+)", line).group(1)
    feature_names = []
    ndarray_list = []
    features = {c_n: [] for c_n in class_names}

    for i in range(0, 1587):
        line = f.readline()
        feature_name = re.match("@attribute (.+) numeric", line)
        if feature_name is not None:
            feature_names.append(feature_name.group(1))

    line = f.readline()
    for i in range(1152):  # 695
        line = f.readline()
        li = line.split(",")
        for class_name in class_names:
            if class_name in li[0]:
                for i in range(1, len(li) - 1):
                    li[i] = float(li[i])
                features[class_name].append(li[1:len(li) - 1])

    for class_name in class_names:
        ndarray_list.append(numpy.vstack(features[class_name]))
    f.close()
    return ndarray_list, class_names, feature_names


def readoneDataFromARFF(arff_filename):
    f = open(arff_filename)
    for i in range(0, 1589):
        f.readline()

    line = f.readline()
    attr_list = line.split(",")
    for i in range(1, len(attr_list) - 1):
        attr_list[i] = float(attr_list[i])
    f.close()
    return numpy.array(attr_list[1:len(attr_list) - 1])
