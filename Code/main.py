import json
import os


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
    jsonOutput = json.dumps(model.transcribe(audio), indent=4)
    open(outputPath, 'w').write(jsonOutput)


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

    print(Summary)
    return Summary


def summarizeGensim(text, compressionRatio):
    from gensim.summarization import summarize
    summary = summarize(text, ratio=compressionRatio)
    print(summary)
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
        print(f"{summaryLUHN}\n")
        return summaryLUHN
    elif "lex" in method:
        lexRankSummarizer = LexRankSummarizer()
        lexRankSummarizerSummary = lexRankSummarizer(parser.document, CompressedSentenceCount)
        summaryLEX = ' '.join([str(sentence) for sentence in lexRankSummarizerSummary])
        print(f"{summaryLEX}\n")
        return summaryLEX
    elif "kl" in method:
        klSummarizer = KLSummarizer()
        klSummarizer = klSummarizer(parser.document, CompressedSentenceCount)
        summaryKL = ' '.join([str(sentence) for sentence in klSummarizer])
        print(f"{summaryKL}\n")
        return summaryKL
    else:
        reductionSummarizer = ReductionSummarizer()
        reductionSummarizerSummary = reductionSummarizer(parser.document, CompressedSentenceCount)
        summaryReduction = ' '.join([str(sentence) for sentence in reductionSummarizerSummary])
        print(f"{summaryReduction}\n")
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
    transcribeVideo(videoPath, 'workingFolder/transcription.json', 'workingFolder')
    summarizedText = loadAndSummarize('workingFolder/transcription.json', compressionRatio, method)
    summarizedSegments = matchAndReturnSegments(summarizedText, 'workingFolder/transcription.json')
    cutAndMergeVideo(videoPath, summarizedSegments, outputPath)
    if cleanFiles:
        os.remove('workingFolder')


summarizeVideo('BlackHoles.mp4', 'SummarizedVideo.mp4', 'sumy-kl', 0.4)
