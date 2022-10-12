import subprocess
import argparse
import pyrosetta
from multiprocessing import Pool, cpu_count
from pathlib import Path

pyrosetta.init()
# pyrosetta.init(options='-constant_seed')

from pyrosetta import *
from pyrosetta.teaching import *  # for energetics
from pyrosetta.rosetta.protocols.relax import *
from pyrosetta.rosetta import core
from pyrosetta.rosetta.core.scoring.methods import EnergyMethodOptions
from tqdm import tqdm

import atom3d.util.formats as fo
import numpy as np
import pandas as pd  # python reads csv faster than xlsx files.

import re
import pickle
import Bio.PDB
import io

def f(mutation):
    subprocess.run(['python', 'process_mutation.py', mutation])

def amino_acids_in_protein_list(sequence):
    return list(sequence)
        
def return_pt_mutations_array(mutations_in_sequence):
    ''' 
    parses mutations string extracted from csv file.
    returns an array, where each array element represents a distinct sequence, each line represented as:
    [str(starting residue identity), res_number, str(ending residue identity)]
    output array dim = [# of sequences, # of pt mutations per sequence, 3]
    '''
    return np.array(re.findall('[^\W\d_]+|\d+', mutations_in_sequence)).reshape(-1, 3)

def mutate_structure(pt_mutations_array, native_structure_to_mutate, repack_radius):
    ''' 
    Performs in silico mutatagensis, returning a new rosettaPose
    Calls pyrosetta.toolbox.mutants.mutate_residue, which takes pose, mutant position, mutant_aa and pack radius
    as inputs.
    '''
    mutant_structure = native_structure_to_mutate.clone()
    for mutate_from, res_number, mutate_to in pt_mutations_array:
        
        res_number = int(res_number)
        if mutate_from != mutant_structure.residue(res_number).name1():
            assert('Residue identity does not match sequence.')
            
        toolbox.mutants.mutate_residue(mutant_structure, res_number, mutate_to, repack_radius)                                       
    return mutant_structure

def init_nonhbond_pairwise_energy_terms_dict(structure, pairwise_rosetta_energy_terms_list):
    return {
        core.scoring.name_from_score_type(scoring_term): np.zeros((structure.total_residue(), structure.total_residue()))
        for scoring_term in pairwise_rosetta_energy_terms_list
    }

def compute_nonhbond_pairwise_residue_energy_terms_dict(structure):
    nonhbond_pairwise_energy_terms_dict = init_nonhbond_pairwise_energy_terms_dict(structure, pairwise_nonhbond_energy_terms_list)
    for res1_idx in range(1, structure.total_residue() + 1):
        res1 = structure.residue(res1_idx)
        for res2_idx in range(res1_idx + 1, structure.total_residue() + 1):
            res2 = structure.residue(res2_idx)
            emap = EMapVector()
            sfxn.eval_ci_2b(res1, res2, structure, emap)
            for energy_term in pairwise_nonhbond_energy_terms_list:
                energy_term_str = core.scoring.name_from_score_type(energy_term)
                nonhbond_pairwise_energy_terms_dict[energy_term_str][res1_idx - 1][res2_idx - 1] = emap[energy_term]
                nonhbond_pairwise_energy_terms_dict[energy_term_str][res2_idx - 1][res1_idx - 1] = emap[energy_term]

    return nonhbond_pairwise_energy_terms_dict
    
def compute_CA_rmsd(structure1, structure2): 
    return rosetta.core.scoring.CA_rmsd(structure1, structure2)

def compute_all_atom_rmsd(structure1, structure2):
    return rosetta.core.scoring.all_atom_rmsd(structure1, structure2)

def relax_structure(structure):
    relaxed_structure = structure.clone()
    relax = FastRelax()
    relax.set_scorefxn(sfxn)
    relax.apply(relaxed_structure)
    return relaxed_structure
    
def compute_intra_residue_energy_terms_dict(structure):
    sfxn = get_score_function(True)
    sfxn(structure)
    
    all_energies_per_residue = pd.DataFrame(structure.energies().residue_total_energies_array())
    energy_term_strs_list = [core.scoring.name_from_score_type(energy_term) 
                             for energy_term in intra_energy_terms_list]
    return all_energies_per_residue[energy_term_strs_list].to_dict('list')

def return_overall_structure_score(structure):
    return structure.energies().active_total_energies()['total_score']

def compute_hbond_energy_terms_dict(structure):
    sfxn = get_score_function(True)
    opts = EnergyMethodOptions()
    opts.hbond_options().decompose_bb_hb_into_pair_energies(True)
    sfxn.set_energy_method_options(opts)
    
    hbond_set = hbonds.HBondSet()
    structure.update_residue_neighbors()
    hbonds.fill_hbond_set(structure, False, hbond_set)
    
    sfxn(structure)  # Note: the residue array will not computed properly if this is not done.
    
    # recompute all energies per residue AFTER redoing energy methods options.
    all_energies_per_residue = pd.DataFrame(structure.energies().residue_total_energies_array())
    
    pairwise_hbond_energies_unweighted_array = np.zeros([structure.total_residue(), structure.total_residue()])
    pairwise_hbond_weights_array = np.zeros([structure.total_residue(), structure.total_residue()])
    
    
    for hbond_idx in range(1, hbond_set.nhbonds() + 1):
        acceptor_res_idx = hbond_set.hbond(hbond_idx).acc_res()
        donor_res_idx = hbond_set.hbond(hbond_idx).don_res()
        unweighted_energy = hbond_set.hbond(hbond_idx).energy()
        weight = hbond_set.hbond(hbond_idx).weight()
        pairwise_hbond_energies_unweighted_array[acceptor_res_idx - 1][donor_res_idx - 1] = unweighted_energy
        pairwise_hbond_energies_unweighted_array[donor_res_idx - 1][acceptor_res_idx - 1] = unweighted_energy
        pairwise_hbond_weights_array[acceptor_res_idx - 1][donor_res_idx - 1] = weight
        pairwise_hbond_weights_array[donor_res_idx - 1][acceptor_res_idx - 1] = weight

    return {'pairwise_hbond_energies_unweighted': pairwise_hbond_energies_unweighted_array,
           'pairwise_hbond_weights': pairwise_hbond_weights_array,
           'per_residue_hbond_sr_bb': all_energies_per_residue['hbond_sr_bb'].to_numpy(),
           'per_residue_hbond_lr_bb': all_energies_per_residue['hbond_lr_bb'].to_numpy(),
           'per_residue_hbond_bb_sc': all_energies_per_residue['hbond_bb_sc'].to_numpy(),
           'per_residue_hbond_sc': all_energies_per_residue['hbond_sc'].to_numpy()}

def all_energetics_list(structure):
    '''
    a list of all energetic terms corresponding to input structure
    '''
    nonhbond_pairwise_energy_terms_dict = compute_nonhbond_pairwise_residue_energy_terms_dict(structure)
    intra_energy_terms_dict = compute_intra_residue_energy_terms_dict(structure)
    non_hbond_overall_weighted_energy = return_overall_structure_score(structure)
    hbond_energy_terms_dict = compute_hbond_energy_terms_dict(structure)
    hbond_overall_weighted_energy = return_overall_structure_score(structure)
    overall_energies_dict = {'non H-bond weighted energy': non_hbond_overall_weighted_energy, 
                             'H-bond weighted energy': hbond_overall_weighted_energy}
    
    return [nonhbond_pairwise_energy_terms_dict,
           intra_energy_terms_dict,
           hbond_energy_terms_dict,
           overall_energies_dict]

def pose_to_dataframe(structure, structure_name):
    '''
    adapted from atom3D class SilentDataset
    '''
    string_stream = rosetta.std.ostringstream()
    structure.dump_pdb(string_stream)
    f = io.StringIO(string_stream.str())
    parser = Bio.PDB.PDBParser(QUIET=True)  # biopython to parse PDB
    bp = parser.get_structure(structure_name, f)
    return fo.bp_to_df(bp)

def mutagenize_and_relax_structure(native_structure_to_mutate, repack_radius, mutations_in_sequence): 
    '''
    returns list of:
    Pose: relaxed_structure
    dataframe: relaxed_structure PDB dumped and parsed
    CA RMSD from native structure
    all atom RMSD from native structure
    '''
    pt_mutations_array = return_pt_mutations_array(mutations_in_sequence)
    structure = mutate_structure(pt_mutations_array, native_structure_to_mutate, repack_radius)
    relaxed_structure = relax_structure(structure)
    return relaxed_structure

def compute_structural_data_dict(relaxed_mutated_structure, native_structure_to_mutate, mutations_in_sequence):
    structure_dataframe = pose_to_dataframe(relaxed_mutated_structure, mutations_in_sequence)
    CA_rmsd = compute_CA_rmsd(relaxed_mutated_structure, native_structure_to_mutate)
    all_atom_rmsd = compute_all_atom_rmsd(relaxed_mutated_structure, native_structure_to_mutate)
    return {'structure dataframe': structure_dataframe,
           'CA RMSD': CA_rmsd,
           'all atom RMSD': all_atom_rmsd}

def structural_data_to_dict(structure_dataframe, CA_rmsd, all_atom_rmsd):
    return {'structure dataframe': structure_dataframe,
           'CA RMSD': CA_rmsd,
           'all atom RMSD': all_atom_rmsd}
    
def mutate_and_compute_features(native_structure_to_mutate, repack_radius, mutations_in_sequence):
    '''
    returns a list of features
    1. nonhbond_pairwise_energy_terms_dict,
    2. intra_energy_terms_dict,
    3. hbond_energy_terms_dict,
    4. overall_energies_dict,
    5. structural_data_dict
    
    and a Pose of the relaxed structure
    '''
    relaxed_mutated_structure = mutagenize_and_relax_structure(native_structure_to_mutate, repack_radius, mutations_in_sequence)
    structural_dict = compute_structural_data_dict(relaxed_mutated_structure, native_structure_to_mutate, mutations_in_sequence)
    features = all_energetics_list(relaxed_mutated_structure)
    features.append(structural_dict)
    return features

def dump_energetics_to_pkl(all_energetics_list, file_name, save_dir):
    pickle.dump(all_energetics_list, open(save_dir + file_name + '.pkl', 'wb'))
    print('Energetic features saved to', file_name + '.pkl')
    
def load_energetics_from_pkl(file_name):
    return pickle.load(open(file_name, 'rb'))

def compute_and_dump_features_for_mutant(native_structure_to_mutate, repack_radius, mutations_in_sequence, save_dir):
    features = mutate_and_compute_features(native_structure_to_mutate, repack_radius, mutations_in_sequence)
    dump_energetics_to_pkl(features, mutations_in_sequence, save_dir)
    print(f'Saved results to {mutations_in_sequence}.pkl')

def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pdb = pose_from_pdb('2gi9_rosetta.pdb')
    native = relax_structure(pdb)
    compute_and_dump_features_for_mutant(native, 10, args.mutation, args.save_dir)

pairwise_nonhbond_energy_terms_list = [fa_atr, fa_rep, fa_sol, lk_ball_wtd, fa_elec]
intra_energy_terms_list = [fa_intra_sol_xover4, pro_close, dslf_fa13, omega,
                           fa_dun, p_aa_pp, yhh_planarity, ref, rama_prepro, total_score]

# save_dir = '/home/t-amywang/rosetta_variability/'
sfxn = get_score_function(True)
parser = argparse.ArgumentParser()
parser.add_argument('mutation')
parser.add_argument('save_dir')
main(parser.parse_args())

