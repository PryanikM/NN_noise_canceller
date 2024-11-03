import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import random
from time import time
from torch.utils.data import random_split
import torch
from torch.utils.data import Dataset, DataLoader
from Hyperparams import Hyperparams as hp


class AudioDenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir):
        if noisy_dir is not None and clean_dir is not None:
            self.noisy_files = self._find_audio_files(noisy_dir)
            self.clean_files = self._find_audio_files(clean_dir)
            # self.clean_files = self.clean_files[:20]

            print(f'{len(self.clean_files) = }')
            print(f'{len(self.noisy_files) = }')
        else:
            self.noisy_files = None
            self.clean_files = None

        self.hop_length = int(hp.sr * hp.frame_shift)
        self.target_samples = int(hp.target_length * hp.sr)

    def __len__(self):
        if self.clean_files is not None:
            return len(self.clean_files)
        else:
            return 0

    def _stft_transform(self, signal):
        # Прямое STFT-преобразование
        stft_output = librosa.stft(signal, n_fft=hp.n_fft, hop_length=self.hop_length, win_length=hp.win_length)
        # Получаем величину спектра
        mag = np.abs(stft_output)  # Форма: (1 + n_fft // 2, T)

        mel_basis = librosa.filters.mel(sr=hp.sr, n_fft=hp.n_fft, n_mels=hp.n_mels)
        mel = np.dot(mel_basis, mag)  # Форма: (n_mels, T)

        # Нормализация и обработка
        mel = 20 * np.log10(np.maximum(1e-5, mel))  # Преобразование в децибелы
        mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)  # Нормализация
        mel = mel.T.astype(np.float32)  # (T, n_mels)

        # Возвращаем mel и mag спектрограммы
        return [torch.tensor(mel, dtype=torch.float32), torch.tensor(mag, dtype=torch.float32)]

    def inverse_stft_transform(self, spectrogram):
        spectrogram = spectrogram.numpy()
        if spectrogram.shape[0] != 1 + hp.n_fft // 2:
            raise ValueError(
                f"Размер спектрограммы не соответствует n_fft. {spectrogram.shape = }   {1 + hp.n_fft // 2 = }")

        reconstructed_audio = librosa.griffinlim(spectrogram, hop_length=self.hop_length, win_length=hp.win_length,
                                                 window="hann")
        return reconstructed_audio

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

        noisy_stft = self._stft_transform(noisy)[1]
        clean_stft = self._stft_transform(clean)[1]
        return noisy_stft.unsqueeze(0), clean_stft.unsqueeze(0)

    def _find_audio_files(self, directory):
        audio_files = []
        audio_extensions = {".mp3", ".wav", ".flac", ".aac", ".ogg"}

        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))

        return audio_files

    def _load_audio(self, file_path):
        audio, _ = librosa.load(file_path, sr=hp.sr)
        if len(audio) > self.target_samples:
            audio = audio[:self.target_samples]
        else:
            padding = self.target_samples - len(audio)
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

    def draw_spectogram(self, *spectrum_with_names):
        num_spectrum = len(spectrum_with_names)
        plt.figure(figsize=(15, 10))

        for i, (spectrum_tensor, name) in enumerate(spectrum_with_names):
            plt.subplot(num_spectrum, 1, i + 1)
            librosa.display.specshow(librosa.amplitude_to_db(spectrum_tensor, ref=np.max), sr=hp.sr,
                                     hop_length=self.hop_length, y_axis='log', x_axis='time')
            plt.title(name)
            plt.colorbar(label='Amplitude')
            plt.xlabel('Time (frames)')
            plt.ylabel('Frequency (bins)')

        plt.tight_layout()
        plt.show()

    def draw_mel_spectogram(self, *spectrum_with_names):
        num_spectrum = len(spectrum_with_names)
        plt.figure(figsize=(15, 10))

        for i, (spectrum_tensor, name) in enumerate(spectrum_with_names):
            plt.subplot(num_spectrum, 1, i + 1)
            librosa.display.specshow(librosa.amplitude_to_db(spectrum_tensor, ref=np.max), sr=hp.sr,
                                     hop_length=self.hop_length, y_axis='mel',
                                     x_axis='time',
                                     cmap='viridis')

            plt.title(name)
            plt.colorbar(label='Amplitude')
            plt.xlabel('Time (frames)')
            plt.ylabel('Frequency (bins)')

        plt.tight_layout()
        plt.show()


class NeuralNetwork:
    def __init__(self, neural_net, optimizer, loss_fn, noisy_dir=None, clean_dir=None, load_model_path=None):
        self.dataset = AudioDenoisingDataset(
            noisy_dir=noisy_dir,
            clean_dir=clean_dir
        )
        if noisy_dir is not None and clean_dir is not None:
            self.train_size = int(hp.train_frac * len(self.dataset))
            self.val_size = len(self.dataset) - self.train_size
            self.train_dataset, self.val_dataset = random_split(self.dataset, [self.train_size, self.val_size])
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=hp.batch_size, shuffle=True)
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=hp.batch_size, shuffle=False)
        else:
            self.train_dataset = None
            self.train_dataloader = None
            self.val_dataloader = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.model = neural_net.to(self.device)

        self.batch_size = hp.batch_size
        self.optimizer = optimizer(self.model.parameters(), lr=hp.lr)
        self.loss_fn = loss_fn
        self.start_epoch = 1
        if load_model_path is not None:
            self.load_checkpoint(load_model_path)

    def get_model(self):
        return self.model

    def get_dataset(self):
        return self.dataset

    def set_model(self, neural_net):
        self.model = neural_net.to(self.device)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1  # Начинаем с следующей эпохи
        # self.loss_fn = checkpoint['loss']

    def study(self, epochs, validation_repeate=0, save_best_model=False, model_path=None):
        if (validation_repeate != 0):
            validation_repeate = validation_repeate
            print_validation = list(range(1, epochs, int(1. / validation_repeate)))
        else:
            print_validation = [0]

        best_val_loss = float('inf')
        loss = 0
        for epoch in range(self.start_epoch, epochs + 1):
            self.model.train()  # Устанавливаем модель в режим обучения
            total_loss = 0
            train_start = time()
            for noisy, clean in self.train_dataloader:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                # Прямой проход
                output = self.model(noisy)

                # Вычисление функции потерь
                loss = self.loss_fn(output, clean)

                # Обратное распространение и оптимизация
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_dataloader)
            if epoch not in print_validation and epoch != epochs:
                train_end = time()
                print(f"Epoch {epoch}/{epochs}, Time = {train_end - train_start}, Train Loss: {avg_train_loss:.4f}")
            else:
                # Валидация
                total_val_loss = 0
                with torch.no_grad():
                    for noisy, clean in self.val_dataloader:
                        noisy, clean = noisy.to(self.device), clean.to(self.device)
                        output = self.model(noisy)
                        loss = self.loss_fn(output, clean)
                        total_val_loss += loss.item()
                train_end = time()
                avg_val_loss = total_val_loss / len(self.val_dataloader)
                print(
                    f"Epoch {epoch}/{epochs}, Time = {train_end - train_start} Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

                if save_best_model and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    if model_path is not None:
                        checkpoint = {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss.item(),
                        }
                        torch.save(checkpoint, f'{model_path}/checkpoint_epoch_{epoch}.pt')
                        print(f"\tCheckpoint saved at epoch {epoch + 1}")
                    print(f"\tBest model saved with Validation Loss: {best_val_loss:.4f}")

    def get_example(self):
        return self.dataset.get_example()

    def get_result_from_model(self, noisy_tensor):
        self.model.eval()
        noisy_tensor = noisy_tensor.unsqueeze(0)
        noisy_tensor = noisy_tensor.to(self.device)
        with torch.no_grad():
            predicted_clean_stft = self.model(noisy_tensor)
        predicted_clean_stft = predicted_clean_stft[0]
        predicted_clean_stft = predicted_clean_stft.cpu()
        clean_signal = self.dataset.inverse_stft_transform(predicted_clean_stft)
        return predicted_clean_stft, clean_signal
