Forced Alignment with Wav2Vec2 and Phonemes

This project demonstrates forced alignment using Wav2Vec2 for aligning phoneme sequences to audio data. It is designed for researchers, linguists, and developers who work with speech processing and phoneme alignment.

Features

Batch Processing: Efficient forced alignment for multiple audio files.

Time-Aligned Segments: Outputs alignment results as JSON files.

Dataset Flexibility: Supports datasets with .wav audio files and .txt phoneme transcriptions in IPA format.

Getting Started

Installation

Clone this repository:

git clone https://github.com/yourusername/forced-alignment.git
cd forced-alignment

Install required dependencies:

pip install -r requirements.txt

Usage

Place your dataset in the required structure:

dataset/
├── wav/            # Audio files (.wav)
├── phonemized/     # Phoneme files (.txt)


Run the alignment script:

python main.py --config config.yaml

Project Structure

.
├── aligner.py          # Core alignment logic
├── dataloader.py       # Dataset handling and preprocessing
├── main.py             # Main script to run the alignment process
├── config.yaml         # Configuration file
├── requirements.txt    # Project dependencies


Output

Aligned segments are saved in JSON format under the specified output folder (segments/ by default). Each JSON file contains the alignment results for its corresponding audio file.

Example Command

Run the alignment process with:

python main.py
