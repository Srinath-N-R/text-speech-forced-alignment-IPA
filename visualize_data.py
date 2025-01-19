import torch
import matplotlib.pyplot as plt

from aligner import NewAligner, get_trellis, constrained_viterbi_alignment, merge_repeats
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
    waveform.squeeze(0).numpy(), 
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
trellis = get_trellis(emission, tokens)

# Perform alignment
path = constrained_viterbi_alignment(emission, tokens)
segments = merge_repeats(path, phonemes)
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

# Plot the emission probabilities
plt.figure(figsize=(10, 6))
plot_emission(emission)
plt.show()

# Plot the trellis
plt.figure(figsize=(10, 6))
plot_trellis(trellis)
plt.show()

# Plot the trellis with path
plt.figure(figsize=(10, 6))
plot_trellis_with_path(trellis, path)
plt.show()

# Plot the alignments
plt.figure(figsize=(10, 6))
plot_alignments(trellis, segments, word_segments, waveform.squeeze(), aligner.sampling_rate)
plt.show()

# Play a segment of the waveform
print("Playing a segment of the audio...")
display_segment(0, waveform, trellis, segments, aligner.sampling_rate)
