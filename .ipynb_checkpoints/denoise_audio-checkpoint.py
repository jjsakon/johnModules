import os
from glob import glob
from scipy.io import wavfile
from scipy.signal import cheb2ord, cheby2, zpk2sos, sosfilt, butter, lfilter
import numpy as np
from multiprocessing import Pool


def butter_bandpass(lowcut, highcut, fs, order=5):
    #return butter(order, [lowcut, highcut], fs=fs, btype='band')
    return butter(order, highcut, fs=fs, btype='low')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def reduce_noise(sample_rate, audio, highcut):
    #data, sr = librosa.load(filepath)

    fp, fs, rp, rs = 200, 5000, 1, 60
    filtered = butter_bandpass_filter(audio, 1, highcut, sample_rate, order=2).astype(np.int16)

    return filtered
#     # filtered = nr.reduce_noise(y=audio, sr=sample_rate)
#     fn = filepath.split("/")[-1]
#     wavfile.write(os.path.join(output_path , fn.replace(".wav", "_denoised.wav")), sample_rate, filtered)


if __name__ == "__main__":
    file = '555_exp_02_preNap_movie_24_audio.wav'
    output_folder = '.'
    reduce_noise(file, output_folder)
    print('Done')
