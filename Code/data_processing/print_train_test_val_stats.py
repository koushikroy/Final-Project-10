import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Define the path to the processed dataset
output_dataset_path = r'C:\Users\rkous\Downloads\Compressed\dataset'

# Define the splits
splits = ['train', 'val', 'test']

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
            actor_id = sample_id.split('-')[-1]
            samples_per_actor[actor_id] += 1
    
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

print("✅ Plots have been saved as 'emotion_distribution_[split].png' and 'actor_distribution_[split].png' in the current directory.")