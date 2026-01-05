import argparse
import os
import numpy as np
import pandas as pd
import glob

def create_split(protein_name, train_frames, data_dir, output_csv):
    """
    Creates a train/val split for the given protein by selecting a continuous
    chunk of frames for training.
    """
    # 1. Find trajectory files
    # Try data_dir/protein_name/*.npy and data_dir/*.npy
    # Normalized search path
    search_patterns = [
        os.path.join(data_dir, protein_name, f"{protein_name}*.npy"),
        os.path.join(data_dir, f"{protein_name}*.npy")
    ]
    
    found_files = []
    for pattern in search_patterns:
        found_files.extend(glob.glob(pattern))
    
    # Remove duplicates if any
    found_files = sorted(list(set(found_files)))
    
    if not found_files:
        print(f"Error: No .npy files found for protein '{protein_name}' in '{data_dir}'")
        return

    print(f"Found {len(found_files)} trajectory file(s) for {protein_name}.")

    # 2. Prepare or Load CSV
    # Columns: name, start_frame, end_frame
    # 'name' here refers to the specific file identifier (e.g. 1a62_A_R1) used by dataset
    fields = ['name', 'train_start', 'train_end', 'total_frames']
    
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv, index_col=None)
    else:
        df = pd.DataFrame(columns=fields)

    new_rows = []

    for fpath in found_files:
        # Extract name (basename without extension)
        # e.g. data/1a62_A/1a62_A_R1.npy -> 1a62_A_R1
        fname = os.path.basename(fpath)
        name_stem = os.path.splitext(fname)[0]

        try:
            # We use mmap_mode='r' to avoid loading the whole file into memory just to get shape
            arr = np.load(fpath, mmap_mode='r')
            total_frames = arr.shape[0]
            
            if total_frames < train_frames:
                print(f"Warning: {name_stem} has {total_frames} frames, which is less than requested train_frames ({train_frames}). Skipping.")
                continue

            # Randomly select start
            # Range: [0, total_frames - train_frames]
            start = np.random.randint(0, total_frames - train_frames + 1)
            end = start + train_frames
            
            print(f"  {name_stem}: Total {total_frames}, Train split [{start}:{end}]")
            
            # Check if this name already exists in DF, if so, update or skip?
            # We'll just append for now, but in a real scenario we might want to overwrite.
            # Let's overwrite if exists to assume a "re-split"
            row = {
                'name': name_stem,
                'train_start': start,
                'train_end': end,
                'total_frames': total_frames
            }
            new_rows.append(row)

        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Filter out old rows for the same names
        df = df[~df['name'].isin(new_df['name'])]
        # Concat
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Save
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Updated splits saved to {output_csv}")
    else:
        print("No valid splits generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val split for a straight protein trajectory.")
    parser.add_argument("name", type=str, help="Protein name (e.g. 1a62_A)")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames for the training split")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory containing .npy files")
    parser.add_argument("--output", type=str, default="gen_model/splits/frame_splits.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    create_split(args.name, args.frames, args.data_dir, args.output)
