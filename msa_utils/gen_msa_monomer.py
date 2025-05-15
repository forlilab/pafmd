import argparse
from pathlib import Path
import os
import subprocess 
import shutil 
import pandas as pd
import itertools
import pickle 
import glob
import sys  
import boto3
from datetime import date
import io 

from msa_processing import format_sto
from msa_helper_functions import list_of_strings, write_fasta_file, get_pdb_w_max_seq_len, get_query_seq_from_msa 

from openfold.data import (
    data_pipeline,
    feature_pipeline,
    mmcif_parsing,
    templates,
    parsers
)
from openfold.data.tools import hhsearch, hmmsearch
from scripts.utils import add_data_args

from custom_openfold_utils.pdb_utils import num_to_chain, get_pdb_id_seq, get_uniprot_id

script_dir = os.path.dirname(os.path.abspath(__file__))

bucket_name = 'openfold'
s3 = boto3.resource(
    service_name='s3',
    region_name='us-east-1'
)
bucket = s3.Bucket(bucket_name)

perl_script = './reformat.pl' 
asterisk_line = '******************************************************************************'

def precompute_alignments(fasta_path, alignment_dir, args):
    template_searcher = hhsearch.HHSearch(
        binary_path=args.hhsearch_binary_path,
        databases=[args.pdb70_database_path],
    )

    alignment_runner = data_pipeline.AlignmentRunner(
        jackhmmer_binary_path=args.jackhmmer_binary_path,
        hhblits_binary_path=args.hhblits_binary_path,
        uniref90_database_path=args.uniref90_database_path,
        mgnify_database_path=args.mgnify_database_path,
        bfd_database_path=args.bfd_database_path,
        uniref30_database_path=args.uniref30_database_path,
        uniclust30_database_path=args.uniclust30_database_path,
        uniprot_database_path=args.uniprot_database_path,
        template_searcher=template_searcher,
        use_small_bfd=args.bfd_database_path is None,
        no_cpus=4
    )

    alignment_runner.run(
        fasta_path, alignment_dir
    )

def generate_features(fasta_path, alignment_dir, args):

    print('GENERATING features.pkl')

    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        max_hits=4,
        kalign_binary_path=args.kalign_binary_path,
        release_dates_path=args.release_dates_path,
        obsolete_pdbs_path=args.obsolete_pdbs_path
    )
    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )
    feature_dict = data_processor.process_fasta(
        fasta_path=fasta_path, alignment_dir=alignment_dir
    )
    features_output_path = os.path.join(alignment_dir, 'features.pkl')
    with open(features_output_path, 'wb') as f:
        pickle.dump(feature_dict, f, protocol=4)
    print('SAVED %s' % features_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template_mmcif_dir", type=str, 
        help="Directory containing mmCIF files to search for templates"
    )
    parser.add_argument(
        "--uniprot_id", type=str, default=None,
        help="",
    )
    parser.add_argument(
        "--pdb_id", type=str, default=None,
        help="Each entry should be of the format XXXX_Y, where XXXX is the pdb_id and Y is the chain_id",
    )
    parser.add_argument(
        "--use_openprotein_alignments", type=bool, default=True
    )
    parser.add_argument(
        "--msa_save_dir", type=str, default=None,
        help="",
    )
    parser.add_argument(
        "--fasta_path", type=str, default=None
    )
    parser.add_argument(
        "--batch_processing", action="store_true", default=False, 
        help="If true, will not prompt the user with I/O"
    )


    add_data_args(parser)
    args = parser.parse_args()

    print(asterisk_line)

    if args.fasta_path:
        #assumes fasta_path is formatted as >protein1_A
        #assumes fasta_path is named as uniprot1.fasta
        with open(args.fasta_path) as fasta_file:
            seqs, tags = parsers.parse_fasta(fasta_file.read())
        pdb_id = tags[0]
        print("Generating alignments for pdb_id/sequence: %s:%s" % (pdb_id,seqs[0]))
        uniprot_str = args.fasta_path.split('/')[-1].split('.')[0]
        msa_dst_dir = '%s/%s/%s' % (args.msa_save_dir, uniprot_str, pdb_id) #subdirectories are inferred from fasta file 
        print("Saving alignments in %s" % msa_dst_dir)
        os.makedirs(msa_dst_dir, exist_ok=True)

        fasta_dst_path = '%s/%s.fasta' % (msa_dst_dir, uniprot_str) 
        shutil.copyfile(args.fasta_path, fasta_dst_path)

        precompute_alignments(args.fasta_path, msa_dst_dir, args)
        generate_features(args.fasta_path, msa_dst_dir, args)
        sys.exit() 

    uniprot_id = args.uniprot_id
    pdb_id = args.pdb_id

    if uniprot_id is None and pdb_id is None:
        raise ValueError("Both uniprot_id and pdb_id cannot be none")
    elif uniprot_id is not None and pdb_id is not None:
        raise ValueError("Both uniprot_id and pdb_id cannot be passed in")
    
    if pdb_id is not None:
        if len(pdb_id.split('_')) != 2:
            raise ValueError("pdb_id must be of format XXXX_Y where XXXX is the pdb_id and Y is the chain_id")

    pdb_openprotein_dict = {} 

    if pdb_id is None:
        #populate corresponding pdbs for each uniprot
        uniprot_pdb_df = pd.read_csv('%s/uniprot_pdb.csv' % script_dir)
        uniprot_pdb_df = uniprot_pdb_df.iloc[1:,]
        uniprot_pdb_df.columns = ['pdb_list']
        uniprot_pdb_df['uniprot_id'] = uniprot_pdb_df.index
        uniprot_pdb_df = uniprot_pdb_df.reset_index(drop=True)
        uniprot_pdb_df = uniprot_pdb_df[['uniprot_id','pdb_list']]

        uniprot_pdb_dict = {} #mapping uniprot_id to all pdb_ids
        pdb_list = list(uniprot_pdb_df[uniprot_pdb_df['uniprot_id'] == uniprot_id]['pdb_list'])[0].split(';')
        uniprot_pdb_dict[uniprot_id] = pdb_list

        print('ALL PDBs for each unique Uniprot ID')
        print(uniprot_pdb_dict)

        openprotein_pdb_dict = {}
        openprotein_pdb_dict[uniprot_id] = [] #mapping uniprot_id to pdb_ids in openprotein 

        if len(uniprot_pdb_dict[uniprot_id]) == 0 or not(args.use_openprotein_alignments):
            rel_pdb_id_w_chain = 'protein0'
        elif len(uniprot_pdb_dict[uniprot_id]) > 0:
            for pdb_id in uniprot_pdb_dict[uniprot_id]:
                prefix = 'pdb/%s' % pdb_id
                objs = list(bucket.objects.filter(Prefix=prefix))
                if len(objs) == 0:
                    continue
                else:
                    curr_pdb_id_w_chain_id = objs[0].key.split('/')[1]
                    curr_pdb_wo_chain_id = curr_pdb_id_w_chain_id.split('_')[0]
                    curr_chain_id = curr_pdb_id_w_chain_id.split('_')[1]
                    if curr_chain_id == 'A':
                        pdb_openprotein_dict[curr_pdb_id_w_chain_id] = 1 
                        openprotein_pdb_dict[uniprot_id].append(curr_pdb_id_w_chain_id)
            if len(openprotein_pdb_dict[uniprot_id]) > 0:          
                #this pdb_id corresponds to pdbs corresponding to uniprot_id present 
                #in openprotein_pdb_dict with the longest sequence 
                rel_pdb_id_w_chain = get_pdb_w_max_seq_len(openprotein_pdb_dict[uniprot_id], uniprot_id)
            else:
                rel_pdb_id_w_chain = 'protein%d' % i
    
        pdb_id = rel_pdb_id_w_chain

    elif uniprot_id is None:
        uniprot_id = get_uniprot_id(pdb_id)
        prefix = 'pdb/%s' % pdb_id
        objs = list(bucket.objects.filter(Prefix=prefix))

        if len(objs) > 0:
            pdb_openprotein_dict[pdb_id] = 1
        else:
            print("pdb_id %s not found in OpenProteinSet" % pdb_id)
            if not(args.batch_processing):
                user_input = input("Do you want to continue? (y/n): If yes, then alignments will be computed from scratch. \n")
                if user_input.lower() == 'n':
                    print("Exiting...")
                    sys.exit()
                elif user_input.lower() == 'y':
                    print("Continuing...")
                else:
                    print("Invalid input. Please enter 'y' to continue or 'n' to exit.")

    print("pdb_id: %s" % pdb_id)
    print('************')
    print("uniprot_id: %s" % uniprot_id)
    print('************')

    if pdb_id not in pdb_openprotein_dict:
        print('MSA for uniprot_id %s not found in OpenProteinSet' % uniprot_id)
        seq = get_pdb_id_seq(pdb_id, uniprot_id) #use sequence corresponding to pdb_id

        print("Generating alignments for pdb_id/sequence: %s:%s" % (pdb_id,seq))
 
        msa_dst_dir = '%s/%s/%s' % (args.msa_save_dir,uniprot_id,pdb_id)
        Path(msa_dst_dir).mkdir(parents=True, exist_ok=True)

        bfd_dst_path = '%s/bfd_uniclust_hits.a3m' % msa_dst_dir 
        mgnify_dst_path = '%s/mgnify_hits.a3m' % msa_dst_dir
        uniref90_dst_path = '%s/uniref90_hits.a3m' % msa_dst_dir
        hhr_dst_path = '%s/pdb70_hits.hhr' % msa_dst_dir

        msa_file_list = [bfd_dst_path,mgnify_dst_path,uniref90_dst_path,hhr_dst_path]
        fasta_path = write_fasta_file([seq], [pdb_id], msa_dst_dir, pdb_id)

        print(msa_file_list)

        if all([os.path.isfile(f) for f in msa_file_list]):
            print('ALIGNMENTS ALREADY PRESENT')
            pass 
        else:
            precompute_alignments(fasta_path, msa_dst_dir, args)
            print('ALIGNMENTS COMPUTED')

    else:
        print('MSA for uniprot_id %s found in OpenProteinSet' % uniprot_id)    
        print("PDB_ID selected: %s" % pdb_id)

        prefix = 'pdb/%s' % pdb_id
        objs = list(bucket.objects.filter(Prefix=prefix))
        
        bfd_src_path = ''
        mgnify_src_path = '' 
        uniref90_src_path = '' 
        hhr_src_path = '' 
        for obj in objs:
            output_fname = '%s/%s/%s' % (args.msa_save_dir, uniprot_id, obj.key)
            obj_path = os.path.dirname(output_fname)
            Path(obj_path).mkdir(parents=True, exist_ok=True)
            bucket.download_file(obj.key, output_fname)
            if 'bfd' in obj.key:
                bfd_src_path = output_fname
            elif 'mgnify' in obj.key:
                mgnify_src_path = output_fname
            elif 'uniref90' in obj.key:
                uniref90_src_path = output_fname
            elif 'hhr' in obj.key:
                hhr_src_path = output_fname 

        print(bfd_src_path)
        print(mgnify_src_path)
        print(uniref90_src_path)
        print(hhr_src_path)

        msa_dst_dir = '%s/%s/%s' % (args.msa_save_dir,uniprot_id,pdb_id)
        Path(msa_dst_dir).mkdir(parents=True, exist_ok=True)

        bfd_dst_path = '%s/bfd_uniclust_hits.a3m' % msa_dst_dir 
        mgnify_dst_path = '%s/mgnify_hits.a3m' % msa_dst_dir
        uniref90_dst_path = '%s/uniref90_hits.a3m' % msa_dst_dir
        hhr_dst_path = '%s/pdb70_hits.hhr' % msa_dst_dir

        if os.path.exists(bfd_src_path):
            shutil.copyfile(bfd_src_path, bfd_dst_path)
        if os.path.exists(mgnify_src_path):
            shutil.copyfile(mgnify_src_path, mgnify_dst_path)
        if os.path.exists(uniref90_src_path):
            shutil.copyfile(uniref90_src_path, uniref90_dst_path)
        if os.path.exists(hhr_src_path):
            shutil.copyfile(hhr_src_path, hhr_dst_path)

        shutil.rmtree('%s/%s/%s' % (args.msa_save_dir, uniprot_id, 'pdb'))

        if os.path.exists(bfd_src_path):
            msa_seq = get_query_seq_from_msa(bfd_dst_path)
        elif os.path.exists(uniref90_dst_path):
            msa_seq = get_query_seq_from_msa(uniref90_dst_path)
            
        fasta_path = write_fasta_file([msa_seq], [pdb_id], msa_dst_dir, pdb_id)


    generate_features(fasta_path, msa_dst_dir, args)
