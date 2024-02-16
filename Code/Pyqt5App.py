import os
from colorama import Fore
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QTimer, QThreadPool, QRunnable, pyqtSlot
from PyQt5.uic import loadUi
import json


def getMatchingPercentages(str1, str2):
    from nltk import word_tokenize
    sentenceOneWords = set(word_tokenize(str1))
    sentenceTwoWords = set(word_tokenize(str2))
    intersection = len(sentenceOneWords.intersection(sentenceTwoWords))
    union = len(sentenceOneWords.union(sentenceTwoWords))
    matching_percentage = (intersection / union) * 100
    return matching_percentage


def transcribeVideo(videoPath, outputPath, workingDir):
    import whisper
    from pydub import AudioSegment
    video = AudioSegment.from_file(videoPath, format="mp4")
    audio = video.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio.export(f"{workingDir}/audio.wav", format="wav")
    model = whisper.load_model("base")
    audio = whisper.load_audio(f'{workingDir}/audio.wav')
    transcribedText = model.transcribe(audio)
    jsonOutput = json.dumps(transcribedText, indent=4)
    open(outputPath, 'w').write(jsonOutput)
    return transcribedText['text']


def summarizeNLTKPure(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    stopwords = set(stopwords.words("english"))
    words = word_tokenize(text)

    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopwords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    sentences = sent_tokenize(text)
    sentenceValue = dict()
    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]
    average = int(sumValues / len(sentenceValue))

    Summary = ''

    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            Summary += " " + sentence

    print(Fore.YELLOW + f'Summary: \n{Summary}' + Fore.RESET)
    return Summary


def summarizeGensim(text, compressionRatio):
    from gensim.summarization import summarize
    summary = summarize(text, ratio=compressionRatio)
    print(Fore.YELLOW + f'Summary: \n{summary}' + Fore.RESET)
    return summary


def summarizeSumy(text, compressionRatio, method):
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.luhn import LuhnSummarizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.kl import KLSummarizer
    from sumy.summarizers.reduction import ReductionSummarizer
    parser = PlaintextParser.from_string(text, Tokenizer('english'))
    CompressedSentenceCount = round(len(parser.tokenize_sentences(text))*compressionRatio)

    if "luhn" in method:
        luhnSummarizer = LuhnSummarizer()
        luhnSummarizerSummary = luhnSummarizer(parser.document, CompressedSentenceCount)
        summaryLUHN = ' '.join([str(sentence) for sentence in luhnSummarizerSummary])
        print(Fore.YELLOW + f"Summary: \n{summaryLUHN}\n" + Fore.RESET)
        return summaryLUHN
    elif "lex" in method:
        lexRankSummarizer = LexRankSummarizer()
        lexRankSummarizerSummary = lexRankSummarizer(parser.document, CompressedSentenceCount)
        summaryLEX = ' '.join([str(sentence) for sentence in lexRankSummarizerSummary])
        print(Fore.YELLOW + f"Summary: \n{summaryLEX}\n" + Fore.RESET)
        return summaryLEX
    elif "kl" in method:
        klSummarizer = KLSummarizer()
        klSummarizer = klSummarizer(parser.document, CompressedSentenceCount)
        summaryKL = ' '.join([str(sentence) for sentence in klSummarizer])
        print(Fore.YELLOW + f"Summary: \n{summaryKL}\n" + Fore.RESET)
        return summaryKL
    else:
        reductionSummarizer = ReductionSummarizer()
        reductionSummarizerSummary = reductionSummarizer(parser.document, CompressedSentenceCount)
        summaryReduction = ' '.join([str(sentence) for sentence in reductionSummarizerSummary])
        print(Fore.YELLOW + f"Summary: \n{summaryReduction}\n" + Fore.RESET)
        return summaryReduction


def loadAndSummarize(transcriptionPath, compressionRatio, method):
    fileContent = json.load(open(transcriptionPath, 'r'))
    if method == "nltk":
        return summarizeNLTKPure(fileContent['text'])
    elif method == "gensim":
        return summarizeGensim(fileContent['text'], compressionRatio)
    elif "sumy" in method:
        return summarizeSumy(fileContent['text'], compressionRatio, method)


def matchAndReturnSegments(summary, transcriptionPath):
    from nltk.tokenize import sent_tokenize
    outSegments = list()
    segmentFile = json.load(open(transcriptionPath, 'r'))
    summarySentences = sent_tokenize(summary)
    for segment in segmentFile['segments']:
        for summarySentence in summarySentences:
            if getMatchingPercentages(segment['text'], summarySentence) > 25:
                if segment not in outSegments:
                    outSegments.append(segment)
    return outSegments


def cutAndMergeVideo(videoPath, segments, outputPath):
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.video.compositing.concatenate import concatenate_videoclips
    videoSegments = []
    for segment in segments:
        videoSegments.append(VideoFileClip(videoPath).subclip(segment['start'], segment['end']))
    summarizedVideo = concatenate_videoclips(videoSegments)
    summarizedVideo.write_videofile(outputPath)


def summarizeVideo(videoPath, outputPath, method, compressionRatio, cleanFiles=False):
    if not os.path.exists('workingFolder'):
        os.mkdir('workingFolder')
    print(Fore.BLUE + "[*] Beginning Text Transcription" + Fore.RESET)
    transcribedText = transcribeVideo(videoPath, 'workingFolder/transcription.json', 'workingFolder')
    print(Fore.GREEN + "[*] Finished Text Transcription" + Fore.RESET)
    print(Fore.BLUE + "[*] Beginning Text Summarization" + Fore.RESET)
    summarizedText = loadAndSummarize('workingFolder/transcription.json', compressionRatio, method)
    print(Fore.GREEN + "[*] Finished Text Summarization" + Fore.RESET)
    print(Fore.BLUE + "[*] Beginning Segment Matching" + Fore.RESET)
    summarizedSegments = matchAndReturnSegments(summarizedText, 'workingFolder/transcription.json')
    print(Fore.GREEN + "[*] Finished Segment Matching" + Fore.RESET)
    print(Fore.BLUE + "[*] Beginning Video Production" + Fore.RESET)
    cutAndMergeVideo(videoPath, summarizedSegments, outputPath)
    print(Fore.GREEN + "[*] Finished Video Production" + Fore.RESET)
    if cleanFiles:
        os.remove('workingFolder')
    return [transcribedText, summarizedText]


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class DataInputDia(QtWidgets.QDialog):
    def __init__(self):
        super(DataInputDia, self).__init__()
        loadUi(resource_path("Resources/DataInput.ui"), self)
        self.setFixedSize(400, 300)
        self.setWindowTitle("Enter Compression Ratio")
        self.setWindowIcon(QtGui.QIcon(resource_path("Resources/logo.png")))

    def getData(self):
        if self.exec_() == QtWidgets.QDialog.Accepted:
            try:
                return float(self.lineEdit.text())
            except Exception:
                return False
        else:
            return False


class MethodDataInputDia(QtWidgets.QDialog):
    def __init__(self):
        super(MethodDataInputDia, self).__init__()
        loadUi(resource_path("Resources/MethodDataInput.ui"), self)
        self.setFixedSize(400, 300)
        self.setWindowTitle("Choose Method")
        self.setWindowIcon(QtGui.QIcon(resource_path("Resources/logo.png")))

    def getData(self):
        if self.exec_() == QtWidgets.QDialog.Accepted:
            try:
                return self.listWidget.currentItem().text()
            except Exception:
                return False
        else:
            return False


class ThreadWorker(QRunnable):
    def __init__(self, inputFilePath, outputFilePath, summarizationMethod, compressionRatio, cleanCode, window):
        super().__init__()
        self.inputFilePath = inputFilePath
        self.outputFilePath = outputFilePath
        self.summarizationMethod = summarizationMethod
        self.compressionRatio = compressionRatio
        self.cleanCode = cleanCode
        self.window = window

    @pyqtSlot()
    def run(self):
        from nltk import sent_tokenize
        [transcribedText, summarizedText] = summarizeVideo(self.inputFilePath,
                                                           self.outputFilePath,
                                                           self.summarizationMethod,
                                                           self.compressionRatio,
                                                           False)
        tempWindowTranscribedText = ""
        tempWindowSummarizedText = ""
        for sentence in sent_tokenize(transcribedText):
            tempWindowTranscribedText += f"{sentence}\n"
        for sentence in sent_tokenize(summarizedText):
            tempWindowSummarizedText += f"{sentence}\n"

        self.window.transcribedText = tempWindowTranscribedText
        self.window.summarizedText = tempWindowSummarizedText


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi(resource_path("Resources/MainScreen.ui"), self)
        self.setFixedSize(800, 600)
        self.setInputBtn.clicked.connect(self.inputClick)
        self.setOutputBtn.clicked.connect(self.outputClick)
        self.summarizeBtn.clicked.connect(self.summarizeClick)
        self.playBtn.clicked.connect(self.playClick)
        self.backBtn.clicked.connect(self.backClick)
        self.setCompressionBtn.clicked.connect(self.compressionClick)
        self.setSummarizationBtn.clicked.connect(self.summarizationClick)
        self.inputFilePath = ""
        self.outputFilePath = ""
        self.compressionRatio = 0.0
        self.summarizationMethod = ""
        self.transcribedText = ""
        self.summarizedText = ""
        self.devCount = 0
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.checkDone)
        self.label_10.mousePressEvent = self.devClick
        self.threadpool = QThreadPool()

    def devClick(self, event):
        self.devCount += 1
        if self.devCount == 7:
            self.successPopup("Done")
            self.devCount = 0
            self.stackedWidget.setCurrentIndex(1)
            self.inputFilePath = 'E:/college/Senior 2/Semester 1/Big Data/Project2/BlackHoles.mp4'
            self.outputFilePath = 'E:/college/Senior 2/Semester 1/Big Data/Project2/o.mp4'
            self.summarizationMethod = "sumy - kl"
            self.compressionRatio = 0.4
            self.summarizeClick()

    def inputClick(self):
        inputFilePath = QtWidgets.QFileDialog.getOpenFileName(filter="Video files(*.mp4)")
        if inputFilePath[0]:
            self.inputFilePath = inputFilePath[0]
            print(Fore.CYAN + f"Input File Set: \n{self.inputFilePath}" + Fore.RESET)

    def outputClick(self):
        outputFilePath = QtWidgets.QFileDialog.getSaveFileName(filter="Video files(*.mp4)")
        if outputFilePath[0]:
            if outputFilePath[0].split(".")[1] != "mp4":
                self.outputFilePath = outputFilePath[0].split(".")[0] + ".mp4"
            else:
                self.outputFilePath = outputFilePath[0]
            print(Fore.CYAN + f"Output File Set: \n{self.outputFilePath}" + Fore.RESET)

    def compressionClick(self):
        inputDia = DataInputDia()
        compressionRatio = inputDia.getData()
        if compressionRatio:
            self.compressionRatio = compressionRatio
            print(Fore.CYAN + f"Compression Ratio Set: {self.compressionRatio}" + Fore.RESET)
        else:
            self.errorPopup("Please enter valid value", "EX: 0.4, 0.5")

    def summarizationClick(self):
        inputDia = MethodDataInputDia()
        summarizationMethod = inputDia.getData()
        if summarizationMethod:
            self.summarizationMethod = str(summarizationMethod).lower()
            print(Fore.CYAN + f"Summarization Method Set: {self.summarizationMethod}" + Fore.RESET)

    def checkDone(self):
        if self.transcribedText and self.summarizedText:
            self.timer.stop()
            self.videoText.setEnabled(True)
            self.summarizeText.setEnabled(True)
            self.playBtn.setEnabled(True)
            self.backBtn.setEnabled(True)
            self.videoText.setText(self.transcribedText)
            self.summarizeText.setText(self.summarizedText)

    def summarizeClick(self):
        if not self.inputFilePath:
            self.errorPopup("Please Set Input File Path!")
            return
        if not self.outputFilePath:
            self.errorPopup("Please Set Output File Path!")
            return
        if not self.compressionRatio:
            self.errorPopup("Please Set Compression Ratio!")
            return
        if not self.summarizationMethod:
            self.errorPopup("Please Set Summarization Method!")
            return
        self.stackedWidget.setCurrentIndex(1)
        self.transcribedText = ""
        self.summarizedText = ""
        try:
            self.timer.start()
            self.videoText.setEnabled(False)
            self.summarizeText.setEnabled(False)
            self.playBtn.setEnabled(False)
            self.backBtn.setEnabled(False)

            summarizationWorker = ThreadWorker(self.inputFilePath,
                                               self.outputFilePath,
                                               self.summarizationMethod,
                                               self.compressionRatio,
                                               False,
                                               self)
            self.threadpool.start(summarizationWorker)

        except Exception:
            self.errorPopup("Something Went Wrong!")

    def playClick(self):
        os.startfile(self.outputFilePath)

    def backClick(self):
        self.stackedWidget.setCurrentIndex(0)

    @staticmethod
    def errorPopup(err_msg, extra=""):
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Error")
        msg.setWindowIcon(QtGui.QIcon(resource_path("Resources/logo.png")))
        msg.setText("An Error Occurred!")
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setInformativeText(err_msg)
        if extra != "": msg.setDetailedText(extra)
        msg.exec_()

    @staticmethod
    def successPopup(success_msg, extra=""):
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Success")
        msg.setWindowIcon(QtGui.QIcon(resource_path("Resources/logo.png")))
        msg.setText("Operation Succeeded\n\n"+success_msg)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        if extra != "": msg.setDetailedText(extra)
        msg.exec_()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.setWindowTitle("VidSummarizer")
    MainWindow.setWindowIcon(QtGui.QIcon(resource_path("Resources/logo.png")))
    MainWindow.show()
    sys.exit(app.exec_())
