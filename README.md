# Forced Alignment with Wav2Vec2 and Phonemes

This project demonstrates forced alignment using Wav2Vec2 for aligning phoneme sequences to audio data.

## Features
- Batch processing for forced alignment.
- Outputs time-aligned segments in JSON format.
- Fully configurable via `config.yaml`.
- Support for datasets with `.wav` and `.txt` phoneme files.

## Setup
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the script:
    ```bash
    python main.py --config config.yaml
    ```

## Project Structure
```plaintext
├── aligner.py          # Core alignment logic
├── dataloader.py       # Dataset handling
├── main.py             # Main script
```plaintext


**## Dataset Format**
wav/: Audio files in .wav format.
phonemized/: Phoneme transcriptions in .txt format.

dataset/
├── wav/
├── phonemized/

**## Output**
Aligned segments are saved as JSON in the segments/ folder.

**## Example Command**
python main.py --config config.yaml
