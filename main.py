import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path

from dataloader import CustomDataset
from aligner import NewAligner


# DATASET should be in this format.
# Phonemized should contain phonemes of transcription in IPA symbols.
# ├── VCTK_train/
# │   ├── wav/            # Audio files (.wav)
# │   ├── phonemized/     # Phoneme files (.txt)

dataset_folder = "/Users/srinathramalingam/Downloads/dataset_new/VCTK_train"
dataset = CustomDataset(dataset_folder)

segments_folder = Path(dataset_folder) / "segments"
segments_folder.mkdir(parents=True, exist_ok=True)

def collate_fn(batch):
    return batch


# Initialize DataLoader
data_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)

aligner = NewAligner()
# Process the data
for batch_idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
    phonemes = [example['phoneme'] for example in batch]
    mod_waveforms = [example['align_audio']['array'] for example in batch]
    mod_waveforms = [torch.tensor(waveform, dtype=torch.float32).to(aligner._device) for waveform in mod_waveforms]
    mod_waveforms = pad_sequence(mod_waveforms, batch_first=True)

    segments = aligner.align_batch(mod_waveforms=mod_waveforms, phonemes=phonemes)

    for i, segment in enumerate(segments):
        audio_path = dataset.audio_files[batch_idx * len(mod_waveforms) + i]
        segment_subfolder = segments_folder / audio_path.parent.name
        segment_subfolder.mkdir(parents=True, exist_ok=True)
        segment_file_path = segment_subfolder / f"{audio_path.stem}.json"
        with open(segment_file_path, "w", encoding="utf-8") as f:
            json.dump(segment, f, indent=4, ensure_ascii=False)
print(f"Segments saved in folder: {segments_folder}")