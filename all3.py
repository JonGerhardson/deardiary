import json
import numpy as np
from pyannote.audio import Inference, Pipeline
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
import joblib
import os
from pyannote.audio import Audio
import torch
import torchaudio
import whisper
import warnings

# Suppress warnings from pyannote.audio
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio")

# NEW: Configuration for speaker registry
SPEAKER_REGISTRY_FILE = "speaker_registry.json"
MIN_SIMILARITY_THRESHOLD = 0.85  # Adjust based on your needs

# Step 1: Set Hugging Face token
os.environ["HUGGINGFACE_TOKEN"] = "no such thing as leftover crack"

# Step 2: Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 3: Load the audio file
audio_file = "/home/jon/deardiary/embeddingstran/untitled.wav"
output_file = "diarized_transcript.txt"
embeddings_file = "speaker_embeddings.json"
print(f"Processing audio file: {audio_file}")

# Step 4: Load the speaker embedding model
embedding_model = Inference(
    "pyannote/embedding",
    use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
).to(device)

# Step 5: Perform speaker diarization
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
).to(device)

print("\nRunning speaker diarization...")
diarization = diarization_pipeline(audio_file)

# Step 6: Extract embeddings for each speaker segment
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

# Step 7: Save embeddings
print(f"\nSaving speaker embeddings to {embeddings_file}...")
with open(embeddings_file, 'w') as f:
    json.dump(segments, f, indent=4)

# Step 8: Speaker Identification and Clustering
print("\nIdentifying known speakers...")

# Load existing speaker registry
registry = {}
try:
    if os.path.exists(SPEAKER_REGISTRY_FILE) and os.path.getsize(SPEAKER_REGISTRY_FILE) > 0:
        with open(SPEAKER_REGISTRY_FILE, 'r') as f:
            registry = json.load(f)
    else:
        # Initialize an empty registry if the file is empty or doesn't exist
        registry = {}
except json.JSONDecodeError:
    print(f"Warning: {SPEAKER_REGISTRY_FILE} contains invalid JSON. Initializing a new registry.")
    registry = {}

# Extract known embeddings and names
known_embeddings = np.array([v['embedding'] for v in registry.values()]) if registry else np.empty((0, 512))
known_names = list(registry.keys()) if registry else []

# Create classifier for known speakers
knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
if len(known_names) > 0:
    knn.fit(known_embeddings, known_names)

# Predict names for segments
predicted_names = []
for emb in np.array(embeddings):
    if len(known_names) > 0:
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
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.7
    ).fit(unknown_embeddings)

    # Assign temporary labels to unknowns
    unknown_labels = clustering.labels_
    temp_names = [f"Unknown_{i+1}" for i in unknown_labels]
else:
    temp_names = []

# Combine known and unknown speaker names
final_names = []
for is_unknown, pred_name, temp_name in zip(unknown_mask, predicted_names, temp_names):
    if is_unknown:
        final_names.append(temp_name)  # Use temporary name for unknown speakers
    else:
        final_names.append(pred_name)  # Use predicted name for known speakers

# Update segments with final names
for i, segment in enumerate(segments):
    segment['speaker'] = final_names[i]

# Save new unknown speakers to registry
print("\nUpdating speaker registry...")
speaker_counts = defaultdict(int)
current_registry = registry.copy()

# Add new embeddings with Unknown_X labels
for name, emb in zip(final_names, embeddings):
    if name.startswith('Unknown_'):
        speaker_counts[name] += 1
        # Register all unknown speakers, regardless of segment count
        if name not in current_registry or speaker_counts[name] > current_registry[name].get('count', 0):
            current_registry[name] = {
                'embedding': emb.tolist(),  # Convert numpy array to list
                'count': speaker_counts[name]
            }
            print(f"Added/updated {name} in registry with {speaker_counts[name]} segments.")

# Save updated registry
with open(SPEAKER_REGISTRY_FILE, 'w') as f:
    json.dump(current_registry, f, indent=4)
print(f"Speaker registry updated with {len(current_registry)} speakers.")

# Step 9: Load Whisper model
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    whisper_model = whisper.load_model("base").to(device)

# Modified preprocessing function
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

# Step 10: Transcribe segments
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

# Step 11: Save results
transcriptions.sort(key=lambda x: x['start'])
with open(output_file, 'w') as f:
    for entry in transcriptions:
        f.write(f"start={entry['start']:.1f}s stop={entry['end']:.1f}s speaker={entry['speaker']}: {entry['text']}\n")

print(f"\nDiarized transcription saved to {output_file}")
