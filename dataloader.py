from pathlib import Path
from datasets import Audio
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.audio_files = sorted(
            [path for path in (Path(dataset_folder) / 'wav').rglob('*.wav') if not path.name.startswith('._')]
        )
        self.phoneme_files = sorted(
            [path for path in (Path(dataset_folder) / 'phonemized').rglob('*.txt') if not path.name.startswith('._')]
        )

        # Get the base file names (without extensions) for matching
        audio_basenames = {path.stem for path in self.audio_files}
        phoneme_basenames = {path.stem for path in self.phoneme_files}

        # Intersection of all file sets (excluding speaker embeddings)
        common_basenames = audio_basenames & phoneme_basenames

        # Filter files to only include common base names
        self.audio_files = [path for path in self.audio_files if path.stem in common_basenames]
        self.phoneme_files = [path for path in self.phoneme_files if path.stem in common_basenames]

        self.audio_feature = Audio(sampling_rate=16000)
    
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = str(self.audio_files[idx])
        phoneme_path = str(self.phoneme_files[idx])
    
        align_audio = self.audio_feature.decode_example({"path": str(audio_path), "bytes": None})

        with open(phoneme_path, 'r') as f:
            phoneme = f.read()
        
        if phoneme is not None:
            phoneme = phoneme.split()
        else:
            phoneme = []

        return {
            'phoneme': phoneme,
            'align_audio': align_audio
        }

