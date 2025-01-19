# Forced Alignment with Wav2Vec2 and Phonemes

This project demonstrates forced alignment using Wav2Vec2 for aligning phoneme sequences to audio data. It is designed for researchers, linguists, and developers who work with speech processing and phoneme alignment.

---

## Features

- **Batch Processing**: Efficient forced alignment for multiple audio files.
- **Time-Aligned Segments**: Outputs alignment results as JSON files.
- **Dataset Flexibility**: Supports datasets with `.wav` audio files and `.txt` phoneme transcriptions in IPA format.
- **Visualization**: Includes tools for visualizing alignment results.

---

## Getting Started

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/forced-alignment.git
   cd forced-alignment
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### Usage

1. Place your dataset in the required structure:
   ```plaintext
   dataset/
   ├── wav/            # Audio files (.wav)
   ├── phonemized/     # Phoneme files (.txt)
   ```

2. Run the alignment script:
   ```bash
   python main.py --config config.yaml
   ```

3. Visualize a single piece of data using the `visualize_data.py` script:
   ```bash
   python visualize_data.py
   ```

---

## Project Structure

```plaintext
.
├── aligner.py          # Core alignment logic
├── dataloader.py       # Dataset handling and preprocessing
├── main.py             # Main script to run the alignment process
├── plot_utils.py       # Utility functions for plotting and visualization
├── visualize_data.py   # Script for visualizing alignment results
├── requirements.txt    # Project dependencies
└── dataset/            # Input dataset folder (user-provided)
```

---

## Dataset Format

The dataset should follow this structure:

```plaintext
dataset/
├── wav/                # Audio files in .wav format
├── phonemized/         # Phoneme transcriptions in .txt format
```

Each `.txt` file should correspond to an audio file and contain phoneme sequences in IPA format.

---

## Output

Aligned segments are saved in JSON format under the specified output folder (`segments/` by default). Each JSON file contains the alignment results for its corresponding audio file.

---

## Example Command

Run the alignment process with:

```bash
python main.py
```

Visualize a single piece of data with:

```bash
python visualize_data.py
```

---
