import argparse
import os
import subprocess
import requests
import zipfile
import sys
# Add project root to path to import scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.prep_sims import traj_to_atom14
from gen_model.split import create_split
import mdtraj
import numpy as np
import tqdm

def download_protein(name, data_dir, redownload=False):
    """Downloads and unzips protein data if not present or if redownload is True."""
    protein_dir = os.path.join(data_dir, name)
    
    if not redownload:
        # Check if directory exists and has content (basic check)
        if os.path.exists(protein_dir) and os.listdir(protein_dir):
            print(f"Data for {name} seems to exist at {protein_dir}. Skipping download.")
            return

    url = f"https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{name}/{name}_protein.zip"
    zip_path = os.path.join(data_dir, f"{name}_protein.zip")
    
    print(f"Downloading {name} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"Failed to download {name}: {e}")
        return

    print(f"Extracting to {protein_dir}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(protein_dir)
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip file.")
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

def preprocess_protein(name, sim_dir, out_dir, suffix='', stride=1, atlas=True, force=False):
    """Runs preprocessing logic similar to prep_sims.py."""
    
    # Create a subdirectory for the protein if it doesn't exist
    protein_out_dir = os.path.join(out_dir, name)
    os.makedirs(protein_out_dir, exist_ok=True)
    
    # Assume 3 replicates for ATLAS data as per prep_sims.py logic
    replicates = [1, 2, 3] if atlas else [1]
    
    for i in replicates:
        output_file = os.path.join(protein_out_dir, f"{name}_R{i}{suffix}.npy") if atlas else os.path.join(protein_out_dir, f"{name}{suffix}.npy")
        
        if not force and os.path.exists(output_file):
            print(f"Preprocessing output {output_file} already exists. Skipping.")
            continue

        print(f"Preprocessing {name} (Replicate {i})...")
        try:
            if atlas:
                traj_path = os.path.join(sim_dir, name, f"{name}_prod_R{i}_fit.xtc")
                pdb_path = os.path.join(sim_dir, name, f"{name}.pdb")
            else:
                # Fallback logic from prep_sims.py for non-atlas execution
                 traj_path = os.path.join(sim_dir, name, f"{name}.xtc")
                 pdb_path = os.path.join(sim_dir, name, f"{name}.pdb")

            # Check if input files exist before loading
            if not os.path.exists(traj_path): 
                 print(f"Missing trajectory file: {traj_path}. Skipping.")
                 continue
            if not os.path.exists(pdb_path):
                 print(f"Missing PDB file: {pdb_path}. Skipping.")
                 continue

            traj = mdtraj.load(traj_path, top=pdb_path)
            
            # Atom slicing logic from prep_sims.py
            if atlas:
                 traj.atom_slice([a.index for a in traj.top.atoms if a.element.symbol != 'H'], True)

            traj.superpose(traj)
            arr = traj_to_atom14(traj)
            np.save(output_file, arr[::stride])
            print(f"Saved to {output_file}")

        except Exception as e:
            print(f"Error processing {name} replicate {i}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess ATLAS protein data.")
    parser.add_argument("name", type=str, help="Protein ID (e.g., 1a62_A)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to download raw data to")
    parser.add_argument("--out_dir", type=str, default="./data", help="Directory to save preprocessed .npy files") # prep_sims default was ./data_atlas, changing to ./data for consistency? checking user pattern
    parser.add_argument("--stride", type=int, default=1, help="Stride for preprocessing")
    parser.add_argument("--train_early_ns", type=float, default=5.0, help="Duration of the early training split in nanoseconds")
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.6, 0.2, 0.2], help="Ratios for train, val, test splits after early part")
    parser.add_argument("--split_output", type=str, default="gen_model/splits/frame_splits.csv", help="Output CSV file for splits")
    parser.add_argument("--redownload", action='store_true', help="Redownload the protein data even if it exists")
    parser.add_argument("--force_prep", action='store_true', help="Force preprocessing even if .npy files exist")
    args = parser.parse_args()

    # Ensure directories exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Download
    download_protein(args.name, args.data_dir, redownload=args.redownload)

    # 2. Preprocess
    # Passing data_dir as sim_dir because download_protein unzips into data_dir/NAME
    # Note: download_atlas.sh unzips into ${name}, effectively creating ./data/${name} if run from root with data_dir=./data
    # But wait, unzip logic in `download_atlas.sh` did `unzip ... -d ${name}`. 
    # If I am in ./data, mkdir ${name}, unzip to ${name}, I get ./data/${name}/${name}_protein/... structure?
    # Let's verify structure. The zip usually contains the files directly or a folder.
    # The shell script was: `mkdir ${name}; unzip ... -d ${name}`. 
    # So raw path will be data_dir/${name}/...
    
    # We need to know the timestep for splitting
    # ATLAS data usually has 10ps per frame. Let's try to detect it.
    ps_per_frame = 10.0 # Default fallback
    try:
        # Load the first two frames of one replicate to get timestep
        folder_name = args.name
        replicate_idx = 1
        traj_path = os.path.join(args.data_dir, folder_name, f"{folder_name}_prod_R{replicate_idx}_fit.xtc")
        pdb_path = os.path.join(args.data_dir, folder_name, f"{folder_name}.pdb")
        if os.path.exists(traj_path) and os.path.exists(pdb_path):
            # Using iterload with chunk=2 to safely get timestep from two frames
            for t in mdtraj.iterload(traj_path, chunk=2, top=pdb_path):
                if t.timestep > 0:
                    ps_per_frame = t.timestep
                    print(f"Detected timestep: {ps_per_frame} ps")
                break
    except Exception as e:
        print(f"Warning: Could not detect timestep, using default {ps_per_frame} ps. Error: {e}")

    # The actual ps_per_frame in the saved .npy depends on stride
    npy_ps_per_frame = ps_per_frame * args.stride

    preprocess_protein(args.name, args.data_dir, args.out_dir, stride=args.stride, atlas=True, force=args.force_prep)

    # 3. Create Split
    print(f"Creating 4-way split (Early: {args.train_early_ns}ns, Ratios: {args.ratios})...")
    create_split(args.name, args.out_dir, args.split_output, 
                 ps_per_frame=npy_ps_per_frame, 
                 train_early_ns=args.train_early_ns, 
                 ratios=tuple(args.ratios))
