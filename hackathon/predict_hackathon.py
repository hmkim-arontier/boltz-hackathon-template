# predict_hackathon.py
import argparse
import json
import os
import sys
from pprint import pprint
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any, List, Optional
import glob
from Bio.PDB import PDBParser, Superimposer, PDBIO
import numpy as np
from itertools import combinations
import yaml
from hackathon_api import Datapoint, Protein, SmallMolecule
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser, SASA, NeighborSearch
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# ---- Participants should modify these four functions ----------------------
# ---------------------------------------------------------------------------

class SpatialEpitopePatchExtractor:
    """
    Extract spatial patches from protein structure based on surface accessibility.
    Each patch represents a potential epitope for antibody binding constraints.
    """
    
    def __init__(self, structure_file):
        """
        Initialize with PDB or CIF structure file.
        
        Args:
            structure_file: Path to PDB or CIF file
        """
        self.structure_file = structure_file
        
        # Determine file type and use appropriate parser
        if structure_file.lower().endswith('.cif'):
            self.parser = MMCIFParser(QUIET=True)
        else:
            self.parser = PDBParser(QUIET=True)
        
        self.structure = self.parser.get_structure('antigen', structure_file)
        self.model = self.structure[0]
        
        # Get all standard residues
        self.residues = [res for res in self.model.get_residues() if res.id[0] == ' ']
        
        # Build neighbor search structure
        self.atoms = [atom for atom in self.model.get_atoms()]
        self.neighbor_search = NeighborSearch(self.atoms)
        
        # Calculate SASA for all residues
        self._calculate_sasa()
        
        # print(f"Loaded structure: {len(self.residues)} residues")
    
    def _calculate_sasa(self):
        """Calculate SASA for all residues using Shrake-Rupley algorithm."""
        print("Calculating SASA...")
        sasa_calculator = SASA.ShrakeRupley()
        sasa_calculator.compute(self.model, level="R")
        print("SASA calculation complete")
    
    def get_residue_center(self, residue):
        """
        Calculate the geometric center of a residue.
        
        Args:
            residue: Bio.PDB residue object
            
        Returns:
            numpy array of 3D coordinates
        """
        coords = np.array([atom.get_coord() for atom in residue.get_atoms()])
        return np.mean(coords, axis=0)
    
    def find_spatial_neighbors(self, residue, radius=10.0):
        """
        Find all residues within spatial distance of target residue.
        
        Args:
            residue: Target residue
            radius: Search radius in Angstroms (default: 10.0)
            
        Returns:
            List of neighboring residues (excluding the target itself)
        """
        center = self.get_residue_center(residue)
        neighbors = self.neighbor_search.search(center, radius, level='R')
        
        # Filter: only standard residues, exclude target residue
        neighbors = [n for n in neighbors if n.id[0] == ' ' and n != residue]
        
        return neighbors
    
    def get_patch_info(self, center_residue, neighbors):
        """
        Extract detailed information about a patch (center + neighbors).
        
        Args:
            center_residue: Central residue of the patch
            neighbors: List of neighboring residues
            
        Returns:
            Dictionary with patch information
        """
        # Get center residue info
        center_chain = center_residue.get_parent().id
        center_resid = center_residue.id[1]
        center_sasa = getattr(center_residue, 'sasa', 0)
        
        # Get neighbor info
        neighbor_info = []
        sasa_values = []
        
        for neighbor in neighbors:
            n_chain = neighbor.get_parent().id
            n_resid = neighbor.id[1]
            n_sasa = getattr(neighbor, 'sasa', 0)
            
            neighbor_info.append({
                'chain': n_chain,
                'resid': n_resid,
                'sasa': n_sasa
            })
            sasa_values.append(n_sasa)
        
        # Calculate mean SASA
        if len(sasa_values) > 0:
            mean_sasa = np.mean(sasa_values)
            total_sasa = np.sum(sasa_values)
        else:
            mean_sasa = 0
            total_sasa = 0
        
        return {
            'center': {
                'chain': center_chain,
                'resid': center_resid,
                'sasa': center_sasa
            },
            'neighbors': neighbor_info,
            'num_neighbors': len(neighbors),
            'mean_sasa': mean_sasa,
            'total_sasa': total_sasa,
            'center_sasa': center_sasa
        }
    
    def extract_all_patches(self, radius=10.0):
        """
        Extract patches for all residues in the structure.
        
        Args:
            radius: Search radius in Angstroms for defining patches
            
        Returns:
            List of patch dictionaries
        """
        print(f"\nExtracting patches (radius={radius}Å)...")
        patches = []
        
        for i, residue in enumerate(self.residues):
            # Find spatial neighbors
            neighbors = self.find_spatial_neighbors(residue, radius)
            
            # Get patch information
            patch_info = self.get_patch_info(residue, neighbors)
            patch_info['patch_id'] = i + 1  # 1-indexed
            
            patches.append(patch_info)
        
        print(f"Extracted {len(patches)} patches")
        return patches
    
    def rank_patches_by_mean_sasa(self, patches):
        """
        Rank patches by mean SASA of neighbors.
        
        Args:
            patches: List of patch dictionaries
            
        Returns:
            List of patches sorted by mean SASA (descending)
        """
        sorted_patches = sorted(patches, key=lambda x: x['mean_sasa'], reverse=True)
        return sorted_patches
    
    def get_top_patches_as_tuples(self, n=20, radius=10.0):
        """
        Get top N patches and return as list of neighbor tuples.
        
        Args:
            n: Number of top patches to return
            radius: Search radius in Angstroms
            
        Returns:
            List of lists, where each inner list contains (chain_id, resid) tuples
            Format: [[(CHID, #), (CHID, #), ...], [(CHID, #), ...], ...]
        """
        # Extract all patches
        patches = self.extract_all_patches(radius=radius)
        
        # Rank by mean SASA
        ranked_patches = self.rank_patches_by_mean_sasa(patches)
        
        # Get top N
        top_patches = ranked_patches[:n]
        
        # Convert to tuple format
        result = []
        for patch in top_patches:
            # Create list of (chain_id, resid) tuples for neighbors
            neighbor_tuples = [
                [neighbor['chain'], neighbor['resid']] 
                for neighbor in patch['neighbors']
            ]
            result.append(neighbor_tuples)
        
        # Display summary
        print("\n" + "="*80)
        print(f"TOP {n} EPITOPE PATCHES (Ranked by Mean SASA)")
        print("="*80)
        
        for i, (patch, neighbor_tuples) in enumerate(zip(top_patches, result), 1):
            print(f"\nPatch {i}:")
            print(f"  Center: {patch['center']['chain']}:{patch['center']['resid']}")
            print(f"  Number of neighbors: {len(neighbor_tuples)}")
            print(f"  Mean neighbor SASA: {patch['mean_sasa']:.2f} Ų")
            print(f"  Neighbors: {neighbor_tuples[:5]}{'...' if len(neighbor_tuples) > 5 else ''}")
        
        return result
        
def get_epitope_patches(cif_path, n_patches=20, radius=10.0):
    """
    Main function to extract epitope patches from a CIF file.
    
    Args:
        cif_path: Path to CIF file
        n_patches: Number of top patches to return (default: 20)
        radius: Search radius in Angstroms (default: 10.0)
        
    Returns:
        neighbours: List of N patches, where each patch is a list of (chain_id, resid) tuples
                   Format: [[(CHID, #), (CHID, #), ...], [(CHID, #), ...], ...]
    """
    extractor = SpatialEpitopePatchExtractor(cif_path)
    neighbours = extractor.get_top_patches_as_tuples(n=n_patches, radius=radius)
    return neighbours

def prepare_protein_complex(datapoint_id: str, proteins: List[Protein], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:
    """
    Prepare input dict and CLI args for a protein complex prediction.
    You can return multiple configurations to run by returning a list of (input_dict, cli_args) tuples.
    Args:
        datapoint_id: The unique identifier for this datapoint
        proteins: List of protein sequences to predict as a complex
        input_dict: Prefilled input dict
        msa_dir: Directory containing MSA files (for computing relative paths)
    Returns:
        List of tuples of (final input dict that will get exported as YAML, list of CLI args). Each tuple represents a separate configuration to run.
    """
    # Please note:
    # `proteins`` will contain 3 chains
    # H,L: heavy and light chain of the Fv or Fab region
    # A: the antigen
    #
    # you can modify input_dict to change the input yaml file going into the prediction, e.g.
    # ```
    # input_dict["constraints"] = [{
    #   "contact": {
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME],
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME]
    #   }
    # }]
    # ```
    #
    # will add contact constraints to the input_dict
    ##################################################
    input_dict_antigen_only = {'sequences': [item for item in input_dict['sequences'] if item['protein']['id'] == 'A'],
                               'version': input_dict['version']}
    with open("yaml_antigen_only.yaml", "w") as f:
        yaml.dump(input_dict_antigen_only, f, sort_keys=False)
    out_dir = args.intermediate_dir / "predictions_ag_only"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz"))
    os.system(f"boltz predict yaml_antigen_only.yaml \
                    --devices 1 --out_dir {out_dir} --cache {cache} --no_kernels --override")
    antigen_cif_path = f"{out_dir}/boltz_results_yaml_antigen_only/predictions/yaml_antigen_only/yaml_antigen_only_model_0.cif"
    
    # get neighbours in format: [N1, N2, ..., N20] where Ni=[(CHID, #), (CHID, #), ..., (CHID, #)]
    neighbours = get_epitope_patches(antigen_cif_path, n_patches=10, radius=8.0)
    
    commands = []
    cli_args = ["--diffusion_samples", "3"]
    for i in range(len(neighbours)):
        input_dict_copy = deepcopy(input_dict)
        input_dict_copy["constraints"] = [{"pocket": {"binder": "H",
                                                      "contacts": neighbours[i]}}]
        assert isinstance(input_dict_copy["constraints"], list)
        input_dict_copy["templates"] = [{"cif": antigen_cif_path,
                                         "chain_id": ["A"],
                                         "template_id": ["A"]}]
        commands.append((input_dict_copy, cli_args))
    ##################################################
    return commands

def prepare_protein_ligand(datapoint_id: str, protein: Protein, ligands: list[SmallMolecule], input_dict: dict, msa_dir: Optional[Path] = None) -> List[tuple[dict, List[str]]]:
    """
    Prepare input dict and CLI args for a protein-ligand prediction.
    You can return multiple configurations to run by returning a list of (input_dict, cli_args) tuples.
    Args:
        datapoint_id: The unique identifier for this datapoint
        protein: The protein sequence
        ligands: A list of a single small molecule ligand object 
        input_dict: Prefilled input dict
        msa_dir: Directory containing MSA files (for computing relative paths)
    Returns:
        List of tuples of (final input dict that will get exported as YAML, list of CLI args). Each tuple represents a separate configuration to run.
    """
    # Please note:
    # `protein` is a single-chain target protein sequence with id A
    # `ligands` contains a single small molecule ligand object with unknown binding sites
    # you can modify input_dict to change the input yaml file going into the prediction, e.g.
    # ```
    # input_dict["constraints"] = [{
    #   "contact": {
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME], 
    #       "token1" : [CHAIN_ID, RES_IDX/ATOM_NAME]
    #   }
    # }]
    # ```
    #
    # will add contact constraints to the input_dict

    # Example: predict 5 structures
    cli_args = ["--diffusion_samples", "5"]
    return [(input_dict, cli_args)]


def extract_chain_atoms(structure, chain_id):
    """Extract CA atoms from a specific chain."""
    atoms = []
    for model in structure:
        if chain_id in [chain.id for chain in model]:
            chain = model[chain_id]
            atoms = [atom for residue in chain for atom in residue if atom.name == 'CA']
            break
    return atoms

def align_structures(pdb_file1, pdb_file2, h_chain='H', l_chain='L', rmsd_threshold=2.0):
    """
    Align two PDB structures based on H and L chains.
    
    Args:
        pdb_file1: Path to first PDB file
        pdb_file2: Path to second PDB file
        h_chain: Chain ID for heavy chain (default 'H')
        l_chain: Chain ID for light chain (default 'L')
        rmsd_threshold: RMSD threshold for considering structures similar (default 2.0 Å)
    
    Returns:
        bool: True if structures are well aligned (RMSD < threshold), False otherwise
    """
    parser = PDBParser(QUIET=True)
    
    # Load structures
    structure1 = parser.get_structure('struct1', pdb_file1)
    structure2 = parser.get_structure('struct2', pdb_file2)
    
    # Extract atoms from both chains
    h_atoms1 = extract_chain_atoms(structure1, h_chain)
    l_atoms1 = extract_chain_atoms(structure1, l_chain)
    h_atoms2 = extract_chain_atoms(structure2, h_chain)
    l_atoms2 = extract_chain_atoms(structure2, l_chain)
    
    # Combine H and L chain atoms
    atoms1 = h_atoms1 + l_atoms1
    atoms2 = h_atoms2 + l_atoms2
    
    # Check if both structures have the same number of atoms
    if len(atoms1) != len(atoms2):
        print(f"Warning: Different number of atoms ({len(atoms1)} vs {len(atoms2)})")
        # Use minimum length for alignment
        min_len = min(len(atoms1), len(atoms2))
        atoms1 = atoms1[:min_len]
        atoms2 = atoms2[:min_len]
    
    if len(atoms1) == 0:
        print("Error: No atoms found for alignment")
        return False
    
    # Perform superimposition
    super_imposer = Superimposer()
    super_imposer.set_atoms(atoms1, atoms2)
    rmsd = super_imposer.rms
    
    print(f"RMSD: {rmsd:.3f} Å")
    
    return rmsd < rmsd_threshold

def check_all_structures_similar(pdb_files, h_chain='H', l_chain='L', rmsd_threshold=2.0):
    """
    Check if all 5 PDB structures are similar by pairwise comparison.
    
    Args:
        pdb_files: List of paths to PDB files
        h_chain: Chain ID for heavy chain
        l_chain: Chain ID for light chain
        rmsd_threshold: RMSD threshold for similarity
    
    Returns:
        bool: True if all structures are mutually similar, False otherwise
    """
    n = len(pdb_files)
    print(f"Comparing {n} structures...\n")
    
    # Compare all pairs
    all_similar = True
    for i, j in combinations(range(n), 2):
        print(f"Comparing {pdb_files[i]} vs {pdb_files[j]}:")
        is_similar = align_structures(pdb_files[i], pdb_files[j], h_chain, l_chain, rmsd_threshold)
        print(f"Result: {'Similar' if is_similar else 'Different'}\n")
        
        if not is_similar:
            all_similar = False
    
    return all_similar




def post_process_protein_complex(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path]) -> List[Path]:
    """
    Return ranked model files for protein complex submission.
    Args:
        datapoint: The original datapoint object
        input_dicts: List of input dictionaries used for predictions (one per config)
        cli_args_list: List of command line arguments used for predictions (one per config)
        prediction_dirs: List of directories containing prediction results (one per config)
    Returns: 
        Sorted pdb file paths that should be used as your submission.
    """
    # Collect all PDBs from all configurations
    all_pdbs = []
    all_logs = []
    for prediction_dir in prediction_dirs:
        pdbs = sorted(prediction_dir.glob(f"{datapoint.datapoint_id}_config_*_model_*.pdb"))
        logs = sorted(prediction_dir.glob(f"confidence_{datapoint.datapoint_id}_config_*_model_*.json"))

        result = check_all_structures_similar(pdbs, h_chain="H", l_chain="L", rmsd_threshold=2.5)
        if result:
            all_pdbs.extend(pdbs)
            all_logs.extend(logs)

    threshold = 0.3
    storage_pair_chains_iptm = []
    for pdb_path, log_path in zip(all_pdbs, all_logs):
        with open(log_path, "r") as f:
            data = json.load(f)
        per_chains_iptm = data["pair_chains_iptm"]["2"]
        value = per_chains_iptm["0"] * per_chains_iptm["1"] * per_chains_iptm["2"]

        if value > threshold:
            storage_pair_chains_iptm.append([pdb_path, value]) # <===
    storage_pair_chains_iptm.sort(key=lambda x: x[-1], reverse=True) # in place
    storage_pdbs = [d[0] for d in storage_pair_chains_iptm]
    return storage_pdbs

def post_process_protein_ligand(datapoint: Datapoint, input_dicts: List[dict[str, Any]], cli_args_list: List[list[str]], prediction_dirs: List[Path]) -> List[Path]:
    """
    Return ranked model files for protein-ligand submission.
    Args:
        datapoint: The original datapoint object
        input_dicts: List of input dictionaries used for predictions (one per config)
        cli_args_list: List of command line arguments used for predictions (one per config)
        prediction_dirs: List of directories containing prediction results (one per config)
    Returns: 
        Sorted pdb file paths that should be used as your submission.
    """
    # Collect all PDBs from all configurations
    all_pdbs = []
    for prediction_dir in prediction_dirs:
        config_pdbs = sorted(prediction_dir.glob(f"{datapoint.datapoint_id}_config_*_model_*.pdb"))
        all_pdbs.extend(config_pdbs)
    
    # Sort all PDBs and return their paths
    all_pdbs = sorted(all_pdbs)
    return all_pdbs

# -----------------------------------------------------------------------------
# ---- End of participant section ---------------------------------------------
# -----------------------------------------------------------------------------


DEFAULT_OUT_DIR = Path("predictions")
DEFAULT_SUBMISSION_DIR = Path("submission")
DEFAULT_INPUTS_DIR = Path("inputs")

ap = argparse.ArgumentParser(
    description="Hackathon scaffold for Boltz predictions",
    epilog="Examples:\n"
            "  Single datapoint: python predict_hackathon.py --input-json examples/specs/example_protein_ligand.json --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate\n"
            "  Multiple datapoints: python predict_hackathon.py --input-jsonl examples/test_dataset.jsonl --msa-dir ./msa --submission-dir submission --intermediate-dir intermediate",
    formatter_class=argparse.RawDescriptionHelpFormatter
)

input_group = ap.add_mutually_exclusive_group(required=True)
input_group.add_argument("--input-json", type=str,
                        help="Path to JSON datapoint for a single datapoint")
input_group.add_argument("--input-jsonl", type=str,
                        help="Path to JSONL file with multiple datapoint definitions")

ap.add_argument("--msa-dir", type=Path,
                help="Directory containing MSA files (for computing relative paths in YAML)")
ap.add_argument("--submission-dir", type=Path, required=False, default=DEFAULT_SUBMISSION_DIR,
                help="Directory to place final submissions")
ap.add_argument("--intermediate-dir", type=Path, required=False, default=Path("hackathon_intermediate"),
                help="Directory to place generated input YAML files and predictions")
ap.add_argument("--group-id", type=str, required=False, default=None,
                help="Group ID to set for submission directory (sets group rw access if specified)")
ap.add_argument("--result-folder", type=Path, required=False, default=None,
                help="Directory to save evaluation results. If set, will automatically run evaluation after predictions.")

args = ap.parse_args()

def _prefill_input_dict(datapoint_id: str, proteins: Iterable[Protein], ligands: Optional[list[SmallMolecule]] = None, msa_dir: Optional[Path] = None) -> dict:
    """
    Prepare input dict for Boltz YAML.
    """
    seqs = []
    for p in proteins:
        if msa_dir and p.msa:
            if Path(p.msa).is_absolute():
                msa_full_path = Path(p.msa)
            else:
                msa_full_path = msa_dir / p.msa
            try:
                msa_relative_path = os.path.relpath(msa_full_path, Path.cwd())
            except ValueError:
                msa_relative_path = str(msa_full_path)
        else:
            msa_relative_path = p.msa
        entry = {
            "protein": {
                "id": p.id,
                "sequence": p.sequence,
                "msa": msa_relative_path
            }
        }
        seqs.append(entry)
    if ligands:
        def _format_ligand(ligand: SmallMolecule) -> dict:
            output =  {
                "ligand": {
                    "id": ligand.id,
                    "smiles": ligand.smiles
                }
            }
            return output
        
        for ligand in ligands:
            seqs.append(_format_ligand(ligand))
    doc = {
        "version": 1,
        "sequences": seqs,
    }
    return doc

def _run_boltz_and_collect(datapoint) -> None:
    """
    New flow: prepare input dict, write yaml, run boltz, post-process, copy submissions.
    """
    out_dir = args.intermediate_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    subdir = args.submission_dir / datapoint.datapoint_id
    subdir.mkdir(parents=True, exist_ok=True)

    # Prepare input dict and CLI args
    base_input_dict = _prefill_input_dict(datapoint.datapoint_id, datapoint.proteins, datapoint.ligands, args.msa_dir)

    if datapoint.task_type == "protein_complex":
        configs = prepare_protein_complex(datapoint.datapoint_id, datapoint.proteins, base_input_dict, args.msa_dir)
    elif datapoint.task_type == "protein_ligand":
        configs = prepare_protein_ligand(datapoint.datapoint_id, datapoint.proteins[0], datapoint.ligands, base_input_dict, args.msa_dir)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")

    # Run boltz for each configuration
    all_input_dicts = []
    all_cli_args = []
    all_pred_subfolders = []
    
    input_dir = args.intermediate_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    
    for config_idx, (input_dict, cli_args) in enumerate(configs):
        # Write input YAML with config index suffix
        yaml_path = input_dir / f"{datapoint.datapoint_id}_config_{config_idx}.yaml"
        with open(yaml_path, "w") as f:
            yaml.safe_dump(input_dict, f, sort_keys=False)

        # Run boltz
        cache = os.environ.get("BOLTZ_CACHE", str(Path.home() / ".boltz"))
        fixed = [
            "boltz", "predict", str(yaml_path),
            "--devices", "1",
            "--out_dir", str(out_dir),
            "--cache", cache,
            "--no_kernels",
            "--output_format", "pdb",
            "--override"
        ]
        cmd = fixed + cli_args
        print(f"Running config {config_idx}:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        # Compute prediction subfolder for this config
        pred_subfolder = out_dir / f"boltz_results_{datapoint.datapoint_id}_config_{config_idx}" / "predictions" / f"{datapoint.datapoint_id}_config_{config_idx}"
        
        all_input_dicts.append(input_dict)
        all_cli_args.append(cli_args)
        all_pred_subfolders.append(pred_subfolder)
        
    # Post-process and copy submissions
    if datapoint.task_type == "protein_complex":
        ranked_files = post_process_protein_complex(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)
    elif datapoint.task_type == "protein_ligand":
        ranked_files = post_process_protein_ligand(datapoint, all_input_dicts, all_cli_args, all_pred_subfolders)
    else:
        raise ValueError(f"Unknown task_type: {datapoint.task_type}")
    
    if not ranked_files:
        raise FileNotFoundError(f"No model files found for {datapoint.datapoint_id}")

    for i, file_path in enumerate(ranked_files[:min(len(ranked_files), 5)]):
        target = subdir / (f"model_{i}.pdb" if file_path.suffix == ".pdb" else f"model_{i}{file_path.suffix}")
        shutil.copy2(file_path, target)
        print(f"Saved: {target}")

    if args.group_id:
        try:
            subprocess.run(["chgrp", "-R", args.group_id, str(subdir)], check=True)
            subprocess.run(["chmod", "-R", "g+rw", str(subdir)], check=True)
        except Exception as e:
            print(f"WARNING: Failed to set group ownership or permissions: {e}")

def _load_datapoint(path: Path):
    """Load JSON datapoint file."""
    with open(path) as f:
        return Datapoint.from_json(f.read())

def _run_evaluation(input_file: str, task_type: str, submission_dir: Path, result_folder: Path):
    """
    Run the appropriate evaluation script based on task type.
    
    Args:
        input_file: Path to the input JSON or JSONL file
        task_type: Either "protein_complex" or "protein_ligand"
        submission_dir: Directory containing prediction submissions
        result_folder: Directory to save evaluation results
    """
    script_dir = Path(__file__).parent
    
    if task_type == "protein_complex":
        eval_script = script_dir / "evaluate_abag.py"
        cmd = [
            "python", str(eval_script),
            "--dataset-file", input_file,
            "--submission-folder", str(submission_dir),
            "--result-folder", str(result_folder)
        ]
    elif task_type == "protein_ligand":
        eval_script = script_dir / "evaluate_asos.py"
        cmd = [
            "python", str(eval_script),
            "--dataset-file", input_file,
            "--submission-folder", str(submission_dir),
            "--result-folder", str(result_folder)
        ]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    print(f"\n{'=' * 80}")
    print(f"Running evaluation for {task_type}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 80}\n")
    
    subprocess.run(cmd, check=True)
    print(f"\nEvaluation complete. Results saved to {result_folder}")

def _process_jsonl(jsonl_path: str, msa_dir: Optional[Path] = None):
    """Process multiple datapoints from a JSONL file."""
    print(f"Processing JSONL file: {jsonl_path}")

    for line_num, line in enumerate(Path(jsonl_path).read_text().splitlines(), 1):
        if not line.strip():
            continue

        print(f"\n--- Processing line {line_num} ---")

        try:
            datapoint = Datapoint.from_json(line)
            _run_boltz_and_collect(datapoint)

        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON on line {line_num}: {e}")
            continue
        except Exception as e:
            print(f"ERROR: Failed to process datapoint on line {line_num}: {e}")
            raise e
            continue

def _process_json(json_path: str, msa_dir: Optional[Path] = None):
    """Process a single datapoint from a JSON file."""
    print(f"Processing JSON file: {json_path}")

    try:
        datapoint = _load_datapoint(Path(json_path))
        _run_boltz_and_collect(datapoint)
    except Exception as e:
        print(f"ERROR: Failed to process datapoint: {e}")
        raise

def main():
    """Main entry point for the hackathon scaffold."""
    # Determine task type from first datapoint for evaluation
    task_type = None
    input_file = None
    
    if args.input_json:
        input_file = args.input_json
        _process_json(args.input_json, args.msa_dir)
        # Get task type from the single datapoint
        try:
            datapoint = _load_datapoint(Path(args.input_json))
            task_type = datapoint.task_type
        except Exception as e:
            print(f"WARNING: Could not determine task type: {e}")
    elif args.input_jsonl:
        input_file = args.input_jsonl
        _process_jsonl(args.input_jsonl, args.msa_dir)
        # Get task type from first datapoint in JSONL
        try:
            with open(args.input_jsonl) as f:
                first_line = f.readline().strip()
                if first_line:
                    first_datapoint = Datapoint.from_json(first_line)
                    task_type = first_datapoint.task_type
        except Exception as e:
            print(f"WARNING: Could not determine task type: {e}")
    
    # Run evaluation if result folder is specified and task type was determined
    if args.result_folder and task_type and input_file:
        try:
            _run_evaluation(input_file, task_type, args.submission_dir, args.result_folder)
        except Exception as e:
            print(f"WARNING: Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
