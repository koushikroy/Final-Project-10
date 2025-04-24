import os

def get_split_subject_ids(dataset_root):
    """
    Traverses dataset_root/{train,val,test}/{emotion}/{sample_id}/
    and collects the unique actor (subject) IDs per split.
    """
    splits = ['train', 'val', 'test']
    split_subjects = {}

    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        subjects = set()

        # Each emotion folder under the split
        for emo in os.listdir(split_dir):
            emo_dir = os.path.join(split_dir, emo)
            if not os.path.isdir(emo_dir):
                continue

            # Each sample directory is named like "02-01-06-01-02-01-12"
            for sample_id in os.listdir(emo_dir):
                # Skip non-directory entries
                sample_dir = os.path.join(emo_dir, sample_id)
                if not os.path.isdir(sample_dir):
                    continue

                # Actor ID is the last element after splitting on '-'
                actor_id = sample_id.split('-')[-1]
                subjects.add(actor_id)

        # Sort for readability
        split_subjects[split] = sorted(subjects)

    return split_subjects

if __name__ == '__main__':
    # ← EDIT this to point at your root dataset folder
    dataset_path = r'C:\Users\rkous\OneDrive - Oklahoma A and M System\courses\Spring 25\Deep Learning\Projects\final_project\Final-Project-10\dataset'

    ids = get_split_subject_ids(dataset_path)
    print("Train subject IDs:", ids['train'])
    print("Validation subject IDs:", ids['val'])
    print("Test subject IDs:", ids['test'])
