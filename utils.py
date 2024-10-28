import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import random
from time import time
from torch.utils.data import random_split
import torch
from torch.utils.data import Dataset, DataLoader


def draw_spectogram(spectrogram, sr=16000, hop_length=256):
    librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()


def visualize_results(*spectrum_with_names, sr=16000, hop_length=512):
    num_spectrum = len(spectrum_with_names)
    plt.figure(figsize=(15, 10))

    for i, (spectrum_tensor, name) in enumerate(spectrum_with_names):
        real_part = spectrum_tensor[0].detach().cpu().numpy()
        imag_part = spectrum_tensor[1].detach().cpu().numpy()
        amplitude = np.sqrt(real_part ** 2 + imag_part ** 2)  # Амплитуда = sqrt(real^2 + imag^2)
        amplitude_log = np.log1p(amplitude)  # log(1 + amplitude) для предотвращения log(0)
        amplitude_log_normalized = amplitude_log / np.max(amplitude_log)
        plt.subplot(num_spectrum, 1, i + 1)
        librosa.display.specshow(amplitude_log_normalized, sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
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
        self.clean_files = self.clean_files[:1000]
        print(f'{len(self.noisy_files) = }')
        print(f'{len(self.clean_files) = }')
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


class NeuralNetwork:
    def __init__(self, noisy_dir, clean_dir, sr, n_fft, hop_length, target_length, train_frac, batch_size, neural_net):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_length = target_length
        self.dataset = AudioDenoisingDataset(
            noisy_dir=noisy_dir,
            clean_dir=clean_dir,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            target_length=target_length,
        )
        self.train_size = int(train_frac * len(self.dataset))
        self.val_size = len(self.dataset) - self.train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [self.train_size, self.val_size])
        self.batch_size = batch_size

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.model = neural_net().to(self.device)

    def get_model(self):
        return self.model

    def set_model(self, neural_net):
        self.model = neural_net().to(self.device)

    def study(self, epochs, loss_fn, optimizer, validation_repeate=0, save_best_model=False, model_path=None):
        if (validation_repeate != 0):
            validation_repeate = validation_repeate
            print_validation = list(range(1, epochs, int(1. / validation_repeate)))
        else:
            print_validation = [0]

        best_val_loss = float('inf')
        for epoch in range(1, epochs + 1):
            self.model.train()  # Устанавливаем модель в режим обучения
            total_loss = 0
            train_start = time()
            for noisy, clean in self.train_dataloader:
                noisy, clean = noisy.to(self.device), clean.to(self.device)

                # Прямой проход
                output = self.model(noisy)

                # Вычисление функции потерь
                loss = loss_fn(output, clean)

                # Обратное распространение и оптимизация
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_dataloader)

            if epoch not in print_validation and epoch != epochs:
                train_end = time()
                print(f"Epoch {epoch}/{epochs}, Time = {train_end - train_start}, Train Loss: {avg_train_loss:.4f}")
            else:
                # Валидация
                self.model.eval()  # Устанавливаем модель в режим оценки
                total_val_loss = 0
                with torch.no_grad():
                    for noisy, clean in self.val_dataloader:
                        noisy, clean = noisy.to(self.device), clean.to(self.device)
                        output = self.model(noisy)
                        loss = loss_fn(output, clean)
                        total_val_loss += loss.item()
                train_end = time()
                avg_val_loss = total_val_loss / len(self.val_dataloader)
                print(
                    f"Epoch {epoch}/{epochs}, Time = {train_end - train_start} Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

                if save_best_model and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    if model_path is not None:
                        torch.save(self.model.state_dict(), model_path)
                    print(f"Best model saved with Validation Loss: {best_val_loss:.4f}")
    def get_example(self):
        return self.dataset.get_example()

    def get_result_from_model(self, noisy_tensor):
        self.model.eval()
        noisy_tensor = noisy_tensor.to(self.device)
        with torch.no_grad():
            predicted_clean_stft = self.model(noisy_tensor)
        clean_signal = inverse_stft_transform(predicted_clean_stft)
        return predicted_clean_stft, clean_signal
