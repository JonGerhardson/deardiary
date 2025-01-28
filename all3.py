import json
import numpy as np
import os
import sys
from datetime import datetime
from pyannote.audio import Inference, Pipeline, Audio
from pyannote.core import Segment
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
import torch
import torchaudio
import whisper
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio")

# Configuration
SPEAKER_REGISTRY_FILE = "speaker_registry.json"
MIN_SIMILARITY_THRESHOLD = 0.85
DEFAULT_AUDIO_PATH = "/home/jon/deardiary/embeddingstran/untitled.wav"
MIN_SEGMENTS_FOR_REGISTRATION = 1

def get_audio_path():
    while True:
        user_input = input(f"Enter audio file path [default: {DEFAULT_AUDIO_PATH}]: ").strip()
        audio_path = user_input if user_input else DEFAULT_AUDIO_PATH
        
        if os.path.exists(audio_path):
            return audio_path
        print(f"\nError: File not found at '{audio_path}'. Please try again.\n")

# Set Hugging Face token
os.environ["HUGGINGFACE_TOKEN"] = "no such thing as leftover crack"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get audio file
audio_file = get_audio_path()
output_file = "diarized_transcript.txt"
embeddings_file = "speaker_embeddings.json"
print(f"\nProcessing audio file: {audio_file}")

# Load models
embedding_model = Inference(
    "pyannote/embedding",
    use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
).to(device)

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
).to(device)

# Perform diarization
print("\nRunning speaker diarization...")
diarization = diarization_pipeline(audio_file)

# Extract embeddings
audio = Audio()
embeddings = []
segments = []

for turn, _, speaker in diarization.itertracks(yield_label=True):
    start_time = turn.start
    end_time = turn.end

    waveform, sample_rate = audio.crop(audio_file, Segment(start_time, end_time))
    waveform = torch.as_tensor(waveform, dtype=torch.float32).to(device)

    embedding = embedding_model({"waveform": waveform, "sample_rate": sample_rate})
    embedding_mean = np.mean(embedding.data, axis=0)

    embeddings.append(embedding_mean)
    segments.append({
        'start': start_time,
        'end': end_time,
        'speaker': speaker,
        'embedding': embedding_mean.tolist()
    })

# Save embeddings
print(f"\nSaving speaker embeddings to {embeddings_file}...")
with open(embeddings_file, 'w') as f:
    json.dump(segments, f, indent=4)

# Load speaker registry
registry = {}
try:
    if os.path.exists(SPEAKER_REGISTRY_FILE):
        with open(SPEAKER_REGISTRY_FILE, 'r') as f:
            registry = json.load(f)
except Exception as e:
    print(f"Error loading registry: {str(e)}, starting fresh")

# Prepare known speakers data
known_embeddings = np.array([v['embedding'] for v in registry.values()]) if registry else np.empty((0, 512))
known_names = list(registry.keys()) if registry else []

# Classify known speakers
knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
if known_names:
    knn.fit(known_embeddings, known_names)

predicted_names = []
for emb in np.array(embeddings):
    if known_names:
        dist, idx = knn.kneighbors([emb], return_distance=True)
        if dist[0][0] <= (1 - MIN_SIMILARITY_THRESHOLD):
            predicted_names.append(known_names[idx[0][0]])
        else:
            predicted_names.append(None)
    else:
        predicted_names.append(None)

# Cluster unknown speakers
unknown_mask = np.array([name is None for name in predicted_names])
unknown_embeddings = np.array(embeddings)[unknown_mask]

if len(unknown_embeddings) > 0:
    normalized_embeddings = unknown_embeddings / np.linalg.norm(unknown_embeddings, axis=1, keepdims=True)
    clustering = DBSCAN(
        eps=0.2,        # Increased from 0.35 to allow more flexibility
        min_samples=2,  # Minimum segments to form a cluster
        metric='cosine'
    ).fit(normalized_embeddings)
    
    unknown_labels = clustering.labels_
    valid_mask = unknown_labels != -1  # Filter out noise points
    unique_labels = np.unique(unknown_labels[valid_mask])
    
    # Assign meaningful names
    temp_names = []
    for label in unknown_labels:
        if label == -1:
            temp_names.append("Unassigned")  # Noise points
        else:
            temp_names.append(f"Unknown_{label + 1}")
else:
    temp_names = []
print(f"\nClustering results:")
print(f"Total segments: {len(unknown_embeddings)}")
print(f"Unique clusters: {len(np.unique(unknown_labels))}")
print(f"Noise points (Unassigned): {np.sum(unknown_labels == -1)}")

# Combine results
final_names = []
temp_idx = 0
for is_unknown in unknown_mask:
    if is_unknown:
        final_names.append(temp_names[temp_idx])
        temp_idx += 1
    else:
        final_names.append(predicted_names[len(final_names)])
        
print("\nSpeaker distribution:")
from collections import Counter
speaker_distribution = Counter(final_names)
for speaker, count in speaker_distribution.items():
    print(f"{speaker}: {count} segments")
    
# Update segments
for i, segment in enumerate(segments):
    segment['speaker'] = final_names[i]

# Update speaker registry
print("\nUpdating speaker registry...")
speaker_counts = defaultdict(int)
current_registry = registry.copy()

for name, emb in zip(final_names, embeddings):
    if name.startswith('Unknown_') and name != "Unassigned":
        speaker_counts[name] += 1
        
        # Only register if speaker meets minimum segment count
        if speaker_counts[name] >= MIN_SEGMENTS_FOR_REGISTRATION:
            existing_entry = current_registry.get(name)
            
            # Update only if new or better than existing
            if not existing_entry or speaker_counts[name] > existing_entry.get('count', 0):
                current_registry[name] = {
                    'embedding': emb.tolist(),
                    'count': speaker_counts[name],
                    'first_seen': datetime.now().isoformat()
                }
                print(f"Registered {name} with {speaker_counts[name]} segments")

# Cleanup transient speakers
current_registry = {
    name: data for name, data in current_registry.items() 
    if data['count'] >= MIN_SEGMENTS_FOR_REGISTRATION
}

# Save registry
try:
    with open(SPEAKER_REGISTRY_FILE, 'w') as f:
        json.dump(current_registry, f, indent=4)
    print(f"Registry updated with {len(current_registry)} valid speakers")
except Exception as e:
    print(f"Error saving registry: {str(e)}")

# Transcribe with Whisper
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    whisper_model = whisper.load_model("large").to(device)

def preprocess_audio_for_whisper(waveform, sample_rate):
    waveform = waveform.to(device)
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000
        ).to(device)
        waveform = resampler(waveform)
    return waveform.unsqueeze(0) if waveform.dim() == 1 else waveform

print("\nTranscribing segments...")
transcriptions = []

for segment in segments:
    try:
        waveform, sample_rate = audio.crop(audio_file, Segment(segment['start'], segment['end']))
        waveform = torch.as_tensor(waveform, dtype=torch.float32).to(device)
        waveform = preprocess_audio_for_whisper(waveform, sample_rate)

        if waveform.numel() == 0:
            print(f"Skipping empty segment: {segment['start']}-{segment['end']}")
            continue

        result = whisper_model.transcribe(
            waveform.squeeze().cpu().numpy(),
            fp16=torch.cuda.is_available(),
            language='en'
        )
        transcriptions.append({
            'start': segment['start'],
            'end': segment['end'],
            'speaker': segment['speaker'],
            'text': result["text"]
        })
    except Exception as e:
        print(f"Error transcribing {segment['start']}-{segment['end']}: {str(e)}")
        transcriptions.append({
            'start': segment['start'],
            'end': segment['end'],
            'speaker': segment['speaker'],
            'text': "[Transcription Error]"
        })

# Save results
transcriptions.sort(key=lambda x: x['start'])
with open(output_file, 'w') as f:
    for entry in transcriptions:
        f.write(f"start={entry['start']:.1f}s stop={entry['end']:.1f}s speaker={entry['speaker']}: {entry['text']}\n")

print(f"\nDiarized transcription saved to {output_file}")
