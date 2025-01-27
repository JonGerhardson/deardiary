# deardiary
diarized transcripts and speaker identification using whisper + pyannote.audio 
## work in progress. don't use this. 

ai gen'd readme. don't read it!

Speaker Diarization and Transcription System

A system for speaker diarization, voice embedding extraction, and audio transcription with speaker identification capabilities. Consists of two main components:

    all3.py - Main processing script for diarization and transcription

    register_speakers.py - Speaker registry management utility

Features

    🎙️ Speaker diarization using pyannote.audio

    🔍 Voice embedding extraction with pyannote/embedding model

    🤖 Automatic speaker clustering for unknown voices

    📝 Speech-to-text transcription using Whisper

    📚 Speaker registry with persistent storage

    👥 Interactive speaker registration/renaming

Installation

    Clone this repository

    Install requirements:

bash
Copy

pip install -r requirements.txt

Required dependencies:

    Python 3.8+

    pyannote.audio 2.0+

    whisper-openai

    numpy

    scikit-learn

    torch

    torchaudio

    huggingface_hub

    Install ffmpeg:

bash
Copy

sudo apt update && sudo apt install ffmpeg

    Accept terms for pyannote models on Hugging Face Hub:

    pyannote/speaker-diarization-3.1

    pyannote/embedding

    Set Hugging Face token in environment:

bash
Copy

export HUGGINGFACE_TOKEN="your_token_here"

Usage

    Process audio file:

bash
Copy

python all3.py

Processes untitled.wav by default. Modify audio_file variable to use different input.

    Register unknown speakers:

bash
Copy

python register_speakers.py

Follow prompts to assign names to unknown speakers detected in the audio.
Configuration

Modify these variables in all3.py:

    audio_file: Path to input audio file (WAV format recommended)

    MIN_SIMILARITY_THRESHOLD: Cosine similarity threshold for speaker matching (0.85 default)

    output_file: Path for transcription output

    embeddings_file: Path for speaker embeddings storage

Environment variables:

    HUGGINGFACE_TOKEN: Your Hugging Face access token

    CUDA_VISIBLE_DEVICES: Configure GPU usage if needed

Speaker Registry Management

The system maintains a speaker_registry.json file that stores:

    Speaker names

    Voice embeddings

    Appearance counts

Workflow:

    Unknown speakers are automatically clustered and labeled (e.g., "Unknown_1")

    Run register_speakers.py to assign permanent names

    Subsequent runs will recognize registered speakers

Output Files

    diarized_transcript.txt: Final transcription with speaker labels

    speaker_embeddings.json: Raw speaker segments with embeddings

    speaker_registry.json: Persistent speaker database

Dependencies Note

Ensure proper CUDA setup for GPU acceleration. The system will automatically use GPU if available. For CPU-only operation, add this to your code:
python
Copy

import torch
torch.set_num_threads(4)  # Adjust based on CPU cores

Limitations

    Real-time processing not supported - optimized for post-processing

    Speaker recognition accuracy depends on audio quality

    Unknown speakers require manual registration

    Whisper model size affects transcription speed/accuracy

Troubleshooting

Common Issues:

    Missing Hugging Face Token:

        Ensure token is set in environment

        Verify token has access to required models

    Empty Speaker Registry:

        Run all3.py first to detect speakers

        Check audio file contains clear speech

    CUDA Out of Memory:

        Use smaller Whisper model

        Reduce audio file duration

        Add torch.cuda.empty_cache() calls

    Invalid Registry JSON:

        Delete corrupted speaker_registry.json

        Reprocess audio file

Example Output

diarized_transcript.txt format:
Copy

start=0.0s stop=4.5s speaker=John: This is the first speech segment
start=4.6s stop=8.2s speaker=Unknown_1: Another speaker not yet registered
start=8.3s stop=12.1s speaker=Sarah: Continuing the conversation

License
https://www.youtube.com/watch?v=QCM0-irFCSk
