import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import random


def find_audio_files(directory):
    audio_files = []
    audio_extensions = {".mp3", ".wav", ".flac", ".aac", ".ogg"}

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))

    return audio_files


def load_audio(files_path, sr=16000, target_length=3.0):
    audios = []
    for file_path in files_path:
        audio, _ = librosa.load(file_path, sr=sr)
        target_samples = int(target_length * sr)

        # Обрезка или дополнение тишиной
        if len(audio) > target_samples:
            audio = audio[:target_samples]  # обрезаем до нужного количества сэмплов
        elif len(audio) < target_samples:
            padding = target_samples - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')  # добавляем тишину
        audios.append(audio)

    return audios


def add_noise(clean_audios, noise_audios):
    mix_audios = []
    for clean_audio in clean_audios:
        noise_factor = np.random.normal(0, 1, clean_audio.shape)
        mix_audios.append(clean_audio + random.choice(noise_audios) * noise_factor)
    return mix_audios


def audio_to_spectrogram(audios, sr=16000, n_fft=512, hop_length=256):
    """Преобразует аудио в спектрограмму."""
    spectrograms = []
    for audio in audios:
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        spectrogram = np.abs(stft)
        spectrograms.append(librosa.amplitude_to_db(spectrogram, ref=np.max))
    return spectrograms


def draw_spectogram(spectrogram, sr=16000, hop_length=256):
    librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()
