import argparse
import os
import numpy as np
import pandas as pd
import glob

def create_split(protein_name, data_dir, output_csv, ps_per_frame=1.0, train_early_ns=5.0, ratios=(0.6, 0.2, 0.2)):
    """
    Creates a contiguous 4-way split (train_early, train, val, test) for the given protein.
    """
    # 1. Find trajectory files
    search_patterns = [
        os.path.join(data_dir, protein_name, f"{protein_name}*.npy"),
        os.path.join(data_dir, f"{protein_name}*.npy")
    ]
    
    found_files = []
    for pattern in search_patterns:
        found_files.extend(glob.glob(pattern))
    
    found_files = sorted(list(set(found_files)))
    
    if not found_files:
        print(f"Error: No .npy files found for protein '{protein_name}' in '{data_dir}'")
        return

    print(f"Found {len(found_files)} trajectory file(s) for {protein_name}.")

    # 2. Prepare or Load CSV
    fields = ['name', 'train_early_end', 'train_end', 'val_end', 'total_frames']
    
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv, index_col=None)
    else:
        df = pd.DataFrame(columns=fields)

    new_rows = []

    for fpath in found_files:
        fname = os.path.basename(fpath)
        name_stem = os.path.splitext(fname)[0]

        try:
            arr = np.load(fpath, mmap_mode='r')
            total_frames = arr.shape[0]
            
            # Calculate early frames (1ns = 1000ps)
            early_frames = int(train_early_ns * 1000 / ps_per_frame)
            early_frames = min(early_frames, total_frames)
            
            remaining_frames = total_frames - early_frames
            
            # Normalize ratios
            r_sum = sum(ratios)
            norm_ratios = [r / r_sum for r in ratios]
            
            train_size = int(remaining_frames * norm_ratios[0])
            val_size = int(remaining_frames * norm_ratios[1])
            # Test size is the rest
            
            train_early_end = early_frames
            train_end = train_early_end + train_size
            val_end = train_end + val_size
            
            print(f"  {name_stem}: Total {total_frames}, Timestep {ps_per_frame}ps")
            print(f"    Early: [0:{train_early_end}], Train: [{train_early_end}:{train_end}], Val: [{train_end}:{val_end}], Test: [{val_end}:{total_frames}]")
            
            row = {
                'name': name_stem,
                'train_early_end': train_early_end,
                'train_end': train_end,
                'val_end': val_end,
                'total_frames': total_frames
            }
            new_rows.append(row)

        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = df[~df['name'].isin(new_df['name'])]
        df = pd.concat([df, new_df], ignore_index=True)
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Updated splits saved to {output_csv}")
    else:
        print("No valid splits generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a contiguous 4-way split for a protein trajectory.")
    parser.add_argument("name", type=str, help="Protein name (e.g. 1a62_A)")
    parser.add_argument("--train_early_ns", type=float, default=5.0, help="Duration of the early training split in nanoseconds")
    parser.add_argument("--ps_per_frame", type=float, default=10.0, help="Picoseconds per frame (timestep * stride)")
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.6, 0.2, 0.2], help="Ratios for train, val, test splits after early part")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory containing .npy files")
    parser.add_argument("--output", type=str, default="gen_model/splits/frame_splits.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    create_split(args.name, args.data_dir, args.output, 
                 ps_per_frame=args.ps_per_frame, 
                 train_early_ns=args.train_early_ns, 
                 ratios=tuple(args.ratios))
