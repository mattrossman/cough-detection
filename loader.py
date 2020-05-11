from praatio import tgio
from scipy.io import wavfile
from glob import glob
import os
import numpy as np
from typing import List, Dict, Tuple
from pydub import AudioSegment
from itertools import tee


sample_rate = 44100


def pairwise(iterable):
    """https://stackoverflow.com/a/5764807"""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class AudioEntry:
    def __init__(self, audio, label):
        self.audio = audio
        self.label = label
    audio: AudioSegment
    label: str


def file_id(path: str) -> str:
    filename = os.path.basename(path)
    return os.path.splitext(filename)[0]


def get_file_ids() -> List[str]:
    return [file_id(path) for path in glob('flusense_audio/*wav')]


def get_labels_path(file: str) -> str:
    return f'flusense_labels/{file}.TextGrid'


def get_audio_path(file: str) -> str:
    return f'flusense_audio/{file}.wav'


def get_audio_entries() -> List[AudioEntry]:
    files = get_file_ids()
    entries = []
    for file in files:
        tg = tgio.openTextgrid(get_labels_path(file))
        audio = AudioSegment.from_wav(get_audio_path(file))
        audio = audio.set_channels(1)  # Convert all to mono
        t_name = tg.tierNameList[0]
        intervals = tg.tierDict[t_name].entryList
        for interval in intervals:
            audio_slice = audio[interval.start * 1000: interval.end * 1000]
            entry = AudioEntry(audio_slice, interval.label)
            entries.append(entry)
    return entries


def partition(pred, iterable):
    trues = []
    falses = []
    for item in iterable:
        if pred(item):
            trues.append(item)
        else:
            falses.append(item)
    return trues, falses


def clip_segment(audio: AudioSegment, duration_ms: int):
    audio = audio[:duration_ms]
    if len(audio) < duration_ms:
        audio = AudioSegment.silent(duration=duration_ms).overlay(audio)
    return audio


def sliding_window_overlap(audio: AudioSegment, window: int):
    """get sliding windows with 50% overlap. window should be in milliseconds"""
    # filter to make sure each window is full
    chunks = filter(lambda x: len(x) == window // 2, audio[::window // 2])
    windows = [a + b for a, b in pairwise(chunks)]  # splice together the halves
    return windows


def sliding_window(audio: AudioSegment, window: int):
    """get sliding windows with no overlap. window should be in milliseconds"""
    # filter to make sure each window is full
    windows = list(audio[::window])
    if len(windows[-1]) < window:
        windows = windows[:-1]
    return windows


def get_data() -> Tuple[List[AudioSegment], List[int]]:
    """Get fixed 1 second recordings and labels (0 = no cough, 1 = cough)"""
    entries = get_audio_entries()
    data = []
    for entry in entries:
        data = data + [(segment, int(entry.label == 'cough')) for segment in sliding_window_overlap(entry.audio, 1000)]
    segments, labels = list(zip(*data))
    return segments, labels


def get_data_small() -> Tuple[List[AudioSegment], List[int]]:
    """Get fixed 1 second recordings and labels (0 = no cough, 1 = cough)"""
    entries = get_audio_entries()
    data = [(clip_segment(entry.audio.set_frame_rate(22050), 1000), int(entry.label == 'cough')) for entry in entries]
    segments, labels = list(zip(*data))
    return segments, labels


def get_numpy_data():
    segments, labels = get_data()
    segments = np.array([seg.get_array_of_samples() for seg in segments])
    labels = np.array(labels)
    return segments, labels


def get_numpy_data_small():
    segments, labels = get_data_small()
    segments = np.array([seg.get_array_of_samples() for seg in segments])
    labels = np.array(labels)
    return segments, labels


def load_audio_data():
    try:
        data = np.load('data/audio.npz')
        return data['arr_0'], data['arr_1']
    except FileNotFoundError:
        print('You need to serialize the data first')


def load_audio_data_small():
    try:
        data = np.load('data/audio-small.npz')
        return data['arr_0'], data['arr_1']
    except FileNotFoundError:
        print('You need to serialize the data first')
