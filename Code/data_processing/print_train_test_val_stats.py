import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Define the path to the processed dataset
output_dataset_path = r'../../dataset'

# Define the splits
splits = ['train', 'val', 'test']

# Data structures to store aggregated data for grouped plots
emotion_distribution = defaultdict(lambda: defaultdict(int))
actor_distribution = defaultdict(lambda: defaultdict(int))

for split in splits:
    split_path = os.path.join(output_dataset_path, split)
    
    total_samples = 0
    samples_per_emotion = defaultdict(int)
    samples_per_actor = defaultdict(int)
    
    # Iterate through emotion folders in the split
    for emotion in os.listdir(split_path):
        emotion_path = os.path.join(split_path, emotion)
        if not os.path.isdir(emotion_path):
            continue
        # Iterate through sample ID folders in the emotion folder
        for sample_id in os.listdir(emotion_path):
            sample_id_path = os.path.join(emotion_path, sample_id)
            if not os.path.isdir(sample_id_path):
                continue
            total_samples += 1
            samples_per_emotion[emotion] += 1
            # Extract actor ID from sample ID (last part after splitting by '-')
            actor_id = sample_id.split('-')[-1].zfill(2)  # Ensure zero-padded actor IDs
            samples_per_actor[actor_id] += 1
    
    # Store data for grouped plots
    for emotion, count in samples_per_emotion.items():
        emotion_distribution[split][emotion] = count
    for actor, count in samples_per_actor.items():
        actor_distribution[split][actor] = count

    # Print statistics for the current split
    print(f"Split: {split}")
    print(f"Total samples: {total_samples}")
    print("Samples per emotion:")
    for emotion in sorted(samples_per_emotion):
        print(f"{emotion}: {samples_per_emotion[emotion]}")
    print("Samples per actor:")
    for actor in sorted(samples_per_actor):
        print(f"Actor {actor}: {samples_per_actor[actor]}")
    print("-" * 40)
    
    # Generate and save plots
    # Plot 1: Bar plot for samples per emotion
    plt.figure(figsize=(10, 6))
    emotions = sorted(samples_per_emotion.keys())
    counts = [samples_per_emotion[emo] for emo in emotions]
    plt.bar(emotions, counts, color='skyblue')
    plt.title(f'Samples per Emotion in {split.capitalize()} Split')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'emotion_distribution_{split}.png')
    plt.close()
    
    # Plot 2: Bar plot for samples per actor
    plt.figure(figsize=(12, 6))
    actors = sorted(samples_per_actor.keys())
    counts = [samples_per_actor[actor] for actor in actors]
    plt.bar(actors, counts, color='lightgreen')
    plt.title(f'Samples per Actor in {split.capitalize()} Split')
    plt.xlabel('Actor ID')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    plt.savefig(f'actor_distribution_{split}.png')
    plt.close()

# Plot 3: Grouped bar chart for emotion distribution across splits
plt.figure(figsize=(12, 6))
emotions = sorted(set(emotion for split_data in emotion_distribution.values() for emotion in split_data))
x = np.arange(len(emotions))
width = 0.25

for i, split in enumerate(splits):
    counts = [emotion_distribution[split][emo] for emo in emotions]
    plt.bar(x + i * width, counts, width, label=split.capitalize())

plt.title('Grouped Emotion Distribution Across Splits')
plt.xlabel('Emotion')
plt.ylabel('Number of Samples')
plt.xticks(x + width, emotions, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('grouped_emotion_distribution.png')
plt.close()

# Plot 4: Combined bar chart for actor distribution across splits
plt.figure(figsize=(14, 8))
actors = [str(i).zfill(2) for i in range(1, 25)]  # Ensure zero-padded actor IDs
x = np.arange(len(actors))
train_counts = [actor_distribution['train'].get(actor, 0) for actor in actors]
val_counts = [actor_distribution['val'].get(actor, 0) for actor in actors]
test_counts = [actor_distribution['test'].get(actor, 0) for actor in actors]

plt.bar(x - 0.2, train_counts, width=0.2, label='Train', color='blue')
plt.bar(x, val_counts, width=0.2, label='Val', color='orange')
plt.bar(x + 0.2, test_counts, width=0.2, label='Test', color='green')

plt.title('Actor Distribution Across Splits')
plt.xlabel('Actor ID')
plt.ylabel('Number of Samples')
plt.xticks(x, actors, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('combined_actor_distribution.png')
plt.close()

print("✅ All plots have been saved, including grouped and combined plots.")