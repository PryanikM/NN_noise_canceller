class Hyperparams:
    # signal processing
    sr = 22050  # Sampling rate.
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.01  # seconds
    frame_length = 0.05  # seconds
    win_length = int(sr * frame_length)  # samples. =1102.
    n_mels = 128  # Number of Mel banks to generate
    max_db = 100
    ref_db = 20
    target_length = 5
    lr=0.001

    # Model scheme
    dropout_rate = 0.05

    # training scheme
    batch_size = 10  # batch size
    epochs = 100
    train_frac = 0.1