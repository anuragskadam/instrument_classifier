import torchaudio
import matplotlib.pyplot as plt

# Load an audio file
audio_path = "archive/Train_submission/Train_submission/rock_8_120BPM.wav"

waveform, sample_rate = torchaudio.load(audio_path)
print(waveform, sample_rate)

# Create a spectrogram transformer
spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=2048, hop_length=512)


# Apply the transform to the waveform
spectrogram = spectrogram_transform(waveform)

# Plot the spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(
    spectrogram.log2()[0, :, :].numpy(), cmap="viridis", aspect="auto", origin="lower"
)
plt.title("Spectrogram")
plt.ylabel("Frequency Bin")
plt.xlabel("Time Frame")
plt.colorbar(format="%+2.0f dB")
plt.show()