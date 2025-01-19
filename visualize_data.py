import torch
import matplotlib.pyplot as plt

from aligner import NewAligner
from dataloader import CustomDataset
from plot_utils import (
    plot_emission,
    plot_trellis,
    plot_trellis_with_path,
    plot_alignments,
    display_segment,
)

# DATASET should be in this format.
# Phonemized should contain phonemes of transcription in IPA symbols.
# ├── VCTK_train/
# │   ├── wav/            # Audio files (.wav)
# │   ├── phonemized/     # Phoneme files (.txt)

dataset_folder = "/Users/srinathramalingam/Downloads/dataset_new/VCTK_train"
dataset = CustomDataset(dataset_folder)

# Load a single piece of data
sample_index = 0  # You can change this index to visualize other samples
sample = dataset[sample_index]
phonemes = sample["phoneme"]
waveform = torch.tensor(sample["align_audio"]["array"]).unsqueeze(0).float()

# Initialize the aligner
aligner = NewAligner()

# Process the data
print("Generating emissions...")
inputs = aligner.processor(
    waveform, 
    sampling_rate=aligner.sampling_rate, 
    return_tensors="pt", 
    padding=True
)
with torch.no_grad():
    emissions = aligner.model(inputs.input_values).logits
emission = emissions[0].cpu()

# Convert phonemes to token IDs and clean them
token_ids = [aligner.tokenizer.convert_tokens_to_ids(p) for p in phonemes]
tokens = torch.tensor(token_ids).to(aligner._device)

# Generate trellis
trellis = aligner.get_trellis(emission, tokens)

# Perform alignment
path = aligner.constrained_viterbi_alignment(emission, tokens)
segments = aligner.merge_repeats(path, phonemes)
ratio = waveform.size(1) / trellis.size(0)
word_segments = [
    {
        "label": seg.label,
        "start": int(seg.start * ratio),
        "end": int(seg.end * ratio),
        "score": seg.score,
    }
    for seg in segments
]

# Visualizations
print("Plotting visualizations...")
plt.figure(figsize=(10, 6))

# Plot the emission probabilities
plt.subplot(2, 2, 1)
plot_emission(emission)

# Plot the trellis
plt.subplot(2, 2, 2)
plot_trellis(trellis)

# Plot the trellis with path
plt.subplot(2, 2, 3)
plot_trellis_with_path(trellis, path)

# Plot the alignments
plt.subplot(2, 2, 4)
plot_alignments(trellis, segments, word_segments, waveform.squeeze(), aligner.sampling_rate)

plt.tight_layout()
plt.show()

# Play a segment of the waveform
print("Playing a segment of the audio...")
display_segment(0, waveform, trellis, segments, aligner.sampling_rate)
