import json
import os

# Configuration
REGISTRY_FILE = "speaker_registry.json"

def load_registry():
    """Load the speaker registry, handling empty or invalid files."""
    if not os.path.exists(REGISTRY_FILE):
        print(f"Registry file {REGISTRY_FILE} does not exist. Please run 'all3.py' first to create it.")
        return None

    try:
        with open(REGISTRY_FILE, 'r') as f:
            if os.path.getsize(REGISTRY_FILE) > 0:
                return json.load(f)
            else:
                print(f"Registry file {REGISTRY_FILE} is empty. Please run 'all3.py' to add speakers.")
                return None
    except json.JSONDecodeError:
        print(f"Error: {REGISTRY_FILE} contains invalid JSON. Please fix or delete the file and run 'all3.py' again.")
        return None

def save_registry(registry):
    """Save the updated registry to the file."""
    try:
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(registry, f, indent=4)
        print(f"Registry saved to {REGISTRY_FILE}.")
    except Exception as e:
        print(f"Error saving registry: {str(e)}")

def register_speakers():
    """Rename unknown speakers in the registry."""
    # Load the registry
    registry = load_registry()
    if registry is None:
        return

    # Find all unknown speakers
    unknown_speakers = {k: v for k, v in registry.items() if k.startswith('Unknown_')}

    if not unknown_speakers:
        print("No unknown speakers found in the registry.")
        print("If you expect unknown speakers, run 'all3.py' to process a new audio file.")
        return

    # Rename unknown speakers
    name_map = {}
    for temp_name, data in unknown_speakers.items():
        while True:
            new_name = input(f"Enter permanent name for {temp_name} (or press Enter to skip): ").strip()
            if new_name:
                if new_name in registry:
                    print(f"Warning: {new_name} already exists in the registry. Choose a different name.")
                else:
                    name_map[temp_name] = new_name
                    break
            else:
                print(f"Skipping {temp_name}.")
                break

    # Update the registry with new names
    for temp_name, new_name in name_map.items():
        registry[new_name] = registry.pop(temp_name)
        print(f"Renamed {temp_name} to {new_name}.")

    # Save the updated registry
    save_registry(registry)
    print(f"\nRegistered {len(name_map)} new speakers!")

if __name__ == "__main__":
    register_speakers()
