# MIT License

# Copyright (c) 2024 Bowen Jing, Hannes Stärk, Tommi Jaakkola, Bonnie Berger

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import mdtraj, os, tqdm
import pandas as pd 
from multiprocessing import Pool
import numpy as np
from gen_model import residue_constants as rc

def traj_to_atom14(traj):
    arr = np.zeros((traj.n_frames, traj.n_residues, 14, 3), dtype=np.float16)
    for i, resi in enumerate(traj.top.residues):
        for at in resi.atoms:
            if at.name not in rc.restype_name_to_atom14_names[resi.name]:
                print(resi.name, at.name, 'not found'); continue
            j = rc.restype_name_to_atom14_names[resi.name].index(at.name)
            arr[:,i,j] = traj.xyz[:,at.index] * 10.0
    return arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='splits/atlas.csv')
    parser.add_argument('--sim_dir', type=str, default='/data/cb/scratch/datasets/atlas')
    parser.add_argument('--outdir', type=str, default='./data_atlas')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--atlas', action='store_true')
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.split, index_col='name')
    names = df.index

    # Define job functions based on args inside main because they depend on args
    if args.atlas:
        def do_job(name):
            for i in [1,2,3]:
                traj = mdtraj.load(f'{args.sim_dir}/{name}/{name}_prod_R{i}_fit.xtc', top=f'{args.sim_dir}/{name}/{name}.pdb') 
                traj.atom_slice([a.index for a in traj.top.atoms if a.element.symbol != 'H'], True)
                traj.superpose(traj)
                arr = traj_to_atom14(traj)
                np.save(f'{args.outdir}/{name}_R{i}{args.suffix}.npy', arr[::args.stride])
    else:
        def do_job(name):
            traj = mdtraj.load(f'{args.sim_dir}/{name}/{name}.xtc', top=f'{args.sim_dir}/{name}/{name}.pdb')
            traj.superpose(traj)
            arr = traj_to_atom14(traj)
            np.save(f'{args.outdir}/{name}{args.suffix}.npy', arr[::args.stride])


    jobs = []
    for name in names:
        if os.path.exists(f'{args.outdir}/{name}{args.suffix}.npy'): continue
        jobs.append(name)

    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    for _ in tqdm.tqdm(__map__(do_job, jobs), total=len(jobs)):
        pass
    if args.num_workers > 1:
        p.__exit__(None, None, None)

if __name__ == "__main__":
    main()