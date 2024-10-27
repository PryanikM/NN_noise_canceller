import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


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


def audio_to_spectrogram(audios, n_fft=1024, hop_length=512):
    """Преобразует аудио в спектрограмму."""
    spectrograms = []
    for audio in audios:
        stft_output = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        stft_complex = np.stack((stft_output.real, stft_output.imag), axis=0)
        spectrograms.append(torch.tensor(stft_complex).unsqueeze(0).float())
    return spectrograms


def spectrogram_to_audio(spectrograms, sr=16000, n_fft=512, hop_length=256):
    audios = []
    for spectrogram in spectrograms:
        audios.append(librosa.istft(spectrogram))
    return audios


def draw_spectogram(spectrogram, sr=16000, hop_length=256):
    librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()


def visualize_results(*spectrum_with_names, sr=16000):

    num_spectrum = len(spectrum_with_names)
    plt.figure(figsize=(15, 10))

    for i, (spectrum_tensor, name) in enumerate(spectrum_with_names):

        real_part = spectrum_tensor[0].detach().cpu().numpy()
        imag_part = spectrum_tensor[1].detach().cpu().numpy()
        amplitude = np.sqrt(real_part ** 2 + imag_part ** 2)  # Амплитуда = sqrt(real^2 + imag^2)
        amplitude_log = np.log1p(amplitude)  # log(1 + amplitude) для предотвращения log(0)
        amplitude_log_normalized = amplitude_log / np.max(amplitude_log)
        plt.subplot(num_spectrum, 1, i + 1)
        plt.imshow(amplitude_log_normalized, aspect='auto', origin='lower', cmap='viridis')
        plt.title(name)
        plt.colorbar(label='Amplitude')
        plt.xlabel('Time (frames)')
        plt.ylabel('Frequency (bins)')



    plt.tight_layout()
    plt.show()

def inverse_stft_transform(stft, hop_length=512):
        stft_numpy = stft.cpu().squeeze(0).numpy()
        stft_complex = stft_numpy[0] + 1j * stft_numpy[1]
        clean_signal = librosa.istft(stft_complex, hop_length=hop_length)
        return clean_signal

class AudioDenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, sr=16000, n_fft=1024, hop_length=512, target_length=3.):

        self.noisy_files = self._find_audio_files(noisy_dir)
        self.clean_files = self._find_audio_files(clean_dir)
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = target_length

    def __len__(self):
        return len(self.clean_files)

    def normalize_spectrogram(self, spectrogram, method='log'):
        real = spectrogram[0]
        imag = spectrogram[1]

        if method == 'minmax':
            # Масштабируем обе части в диапазон [0, 1]
            real = (real - real.min()) / (real.max() - real.min())
            imag = (imag - imag.min()) / (imag.max() - imag.min())

        elif method == 'log':
            # Логарифмическое преобразование для обеих частей
            real = np.log1p(np.abs(real)) * np.sign(real)
            imag = np.log1p(np.abs(imag)) * np.sign(imag)

        else:
            raise ValueError("Метод нормализации должен быть 'minmax' или 'log'")

        # Объединяем нормализованные действительную и мнимую части обратно
        normalized_stft = np.stack((real, imag), axis=0)
        return normalized_stft

    def _stft_transform(self, signal):
        stft_output = librosa.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length)
        stft_complex = np.stack((stft_output.real, stft_output.imag), axis=0)
        # stft_complex_normalize = self.normalize_spectrogram(stft_complex)
        return torch.tensor(stft_complex, dtype=torch.float32)

    def inverse_stft_transform(self, signal):
        predicted_clean_stft_complex = signal[0] + 1j * signal[1]
        clean_signal = librosa.istft(predicted_clean_stft_complex, hop_length=self.hop_length)
        return clean_signal

    def get_example(self):
        noisy_path = np.random.choice(self.noisy_files)
        clean_path = np.random.choice(self.clean_files)

        noisy_audio = self._load_audio(noisy_path)
        clean_audio = self._load_audio(clean_path)
        mix_audio = self.add_noise(clean_audio, noisy_audio)

        mix_spector = self._stft_transform(mix_audio)
        clean_spector = self._stft_transform(clean_audio)
        noisy_spector = self._stft_transform(noisy_audio)

        return mix_spector, clean_spector, noisy_spector, clean_audio, noisy_audio, mix_audio

    def __getitem__(self, idx):

        noisy_path = np.random.choice(self.noisy_files)
        clean_path = self.clean_files[idx]

        noisy_audio = self._load_audio(noisy_path)
        clean = self._load_audio(clean_path)
        noisy = self.add_noise(clean, noisy_audio)

        noisy_tensor = self._stft_transform(noisy)
        clean_tensor = self._stft_transform(clean)


        return noisy_tensor, clean_tensor

    def _find_audio_files(self, directory):
        audio_files = []
        audio_extensions = {".mp3", ".wav", ".flac", ".aac", ".ogg"}

        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))

        return audio_files

    def _load_audio(self, file_path):
        audio, _ = librosa.load(file_path, sr=self.sr)
        target_samples = int(self.target_length * self.sr)

        # Обрезка или дополнение тишиной
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            padding = target_samples - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        return audio


    def add_noise(self, clean_audio, noise_audio, min_noise_level=0.1, max_noise_level=0.8):
        target_noise_level = random.uniform(min_noise_level, max_noise_level)

        noise_db = 20 * np.log10(np.max(np.abs(noise_audio)))
        clean_db = 20 * np.log10(np.max(np.abs(clean_audio)))

        required_noise_db = clean_db + 10 * np.log10(target_noise_level)
        gain = 10 ** ((required_noise_db - noise_db) / 20)

        mixed_sound = clean_audio + gain * noise_audio
        return mixed_sound


class DenoisingNet(nn.Module):
    def __init__(self):
        super(DenoisingNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=(3, 3), padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
