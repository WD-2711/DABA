from pydub import AudioSegment
import glob

## set path
# import sys, os
# ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
# sys.path.append(ROOT)

def trans_mp3_to_wav(filepath):
    song = AudioSegment.from_mp3(filepath)
    targetpath = filepath[:-4] + ".wav"
    print(targetpath)
    song.export(targetpath, format = "wav")

if __name__ == "__main__":
    path = "./DABA_demo/data/trigger_pool"

    files = glob.glob(path + '/*/*.mp3')
    for f in files:
        trans_mp3_to_wav(f)
