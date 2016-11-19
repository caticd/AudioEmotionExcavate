from models import Files, SqlSession
from Emotion.audioTrainTest import fileClassification
from Emotion.ConfusionMatrix import ConfusionMatrix
import os
import subprocess
import time


PROJECT_DIR = "C:\\Users\\ZYC\\PycharmProjects\\AudioEmotionExcavate"
ALGO_NAMES = ("svm", "randomforest", "gradientboosting", "extratrees")
emotion_list = ("angry", "fear", "happy", "neutral", "sad", "surprise")
svm_confusion_matrix = ConfusionMatrix([[16.7, 0, 1.3, 1.2, 0, 0.8],
                                        [0.5, 10.2, 0.7, 0.3, 2, 0.2],
                                        [2.1, 0.4, 11.5, 1.3, 0.2, 1.1],
                                        [0.7, 0.5, 0.8, 16.3, 0.7, 0.1],
                                        [1, 2.2, 0.5, 1.1, 10.9, 0],
                                        [1.3, 0.2, 1.1, 0.3, 0.3, 11.5]])
randomforest_confusion_matrix = ConfusionMatrix([[15.1, 0, 1.8, 1.7, 0.2, 1.3],
                                                 [0.6, 7.6, 0.7, 1, 3.1, 1],
                                                 [2, 0.1, 10.5, 2, 0, 1.9],
                                                 [0.7, 0.5, 0.5, 16.3, 0.7, 0.3],
                                                 [1, 2.7, 0.4, 1.7, 9.9, 0],
                                                 [1.3, 0.1, 1.6, 0.6, 0.1, 11.1]])
extratrees_confusion_matrix = ConfusionMatrix([[15, 0, 1.9, 1.3, 0.2, 1.6],
                                               [0.7, 7.6, 0.8, 0.6, 3.4, 0.8],
                                               [2.2, 0.1, 10.7, 1.3, 0, 2.2],
                                               [0.9, 0.4, 0.6, 16.5, 0.6, 0.2],
                                               [1.1, 2.8, 0.2, 1.6, 10, 0],
                                               [1, 0.2, 1.4, 0.5, 0.1, 11.6]])
gradientboosting_confusion_matrix = ConfusionMatrix([[16.9, 0.4, 2.2, 2.1, 0.3, 1.3],
                                                     [1.4, 3.8, 1, 0.8, 3.6, 1],
                                                     [2.9, 0.6, 9.4, 1.8, 0.4, 0.8],
                                                     [1.3, 0.6, 1.2, 16.9, 1, 0.8],
                                                     [1.8, 1.8, 1.4, 1.3, 8, 0.3],
                                                     [1.2, 0.3, 1.1, 1.2, 0.2, 9.1]])
confusion_matrixs = {"svm": svm_confusion_matrix,
                     "randomforest": randomforest_confusion_matrix,
                     "extratrees": extratrees_confusion_matrix,
                     "gradientboosting": gradientboosting_confusion_matrix}


def extract_feature():
    sql_session = SqlSession()
    samples = sql_session.query(Files).filter(Files.progress == '1').all()
    for sample in samples:
        sample.progress = 2
        sql_session.commit()
        try:
            file_name = sample.store_name
            os.chdir(PROJECT_DIR)
            result = subprocess.call(
                [".\\smile\\SMILExtract_Release.exe", "-C", ".\\smile\\config\\emobase2010.conf", "-I",
                 ".\\upload_dir\\" + file_name, "-instname", file_name, "-O",
                 ".\\smile\\onearff\\" + file_name[0:len(file_name) - 4] + ".arff"])
        except:
            sample.progress = 3
            sql_session.commit()
        else:
            if result == 0:
                sample.progress = 4
                sql_session.commit()
            else:
                sample.progress = 3
                sql_session.commit()
    if samples:
        return True
    else:
        return False


def classification():
    sql_session = SqlSession()
    samples = sql_session.query(Files).filter(Files.progress == '4').all()
    for sample in samples:
        sample.progress = 5
        sql_session.commit()
        try:
            possible = [0, 0, 0, 0, 0, 0]
            file_name = sample.store_name
            for algorithm in ALGO_NAMES:
                result, p, _ = fileClassification(
                    PROJECT_DIR + "\\smile\\onearff\\" + file_name[:len(file_name) - 4] + ".arff",
                    PROJECT_DIR + "\\Emotion\\" + algorithm + "_result",
                    algorithm)
                for i in range(len(possible)):
                    possible[i] += confusion_matrixs[algorithm][result][i]
        except:
            sample.progress = 6
            sql_session.commit()
        else:
            sample.result = emotion_list[find_max_index(possible)]
            sample.possible_angry = possible[0]
            sample.possible_fear = possible[1]
            sample.possible_happy = possible[2]
            sample.possible_neutral = possible[3]
            sample.possible_sad = possible[4]
            sample.possible_surprise = possible[5]
            sample.progress = 7
            sql_session.commit()
    if samples:
        return True
    else:
        return False


def find_max_index(tuple):
    temp = tuple[0]
    index = 0
    for i in range(1, len(tuple)):
        if tuple[i] > temp:
            index = i
            temp = tuple[i]
    return index


def main():
    while True:
        try:
            while True:
                extract_feature()
                if classification():
                    continue
                time.sleep(10)
        except:
            pass


if __name__ == '__main__':
    main()
