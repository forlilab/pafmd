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
import json 

from msa_processing import format_sto 
from msa_helper_functions import list_of_strings, write_fasta_file, get_pdb_w_max_seq_len, get_query_seq_from_msa

sys.path.insert(0,'../')
from openfold.data import (
    data_pipeline,
    feature_pipeline,
    mmcif_parsing,
    templates,
    parsers
)
from openfold.data.tools import hmmsearch
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

    with open(fasta_path) as fasta_file:
        seqs, tags = parsers.parse_fasta(fasta_file.read())

    seq_alignment_dir_dict = {}
   
    for tag, seq in zip(tags, seqs): 
        tmp_fasta_path = os.path.join(alignment_dir, f"tmp_{tag}.fasta")
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(
            alignment_dir, tag
        )

        if seq not in seq_alignment_dir_dict:
            os.makedirs(local_alignment_dir, exist_ok=True)
            seq_alignment_dir_dict[seq] = local_alignment_dir

            template_searcher = hmmsearch.Hmmsearch(
                binary_path=args.hmmsearch_binary_path,
                hmmbuild_binary_path=args.hmmbuild_binary_path,
                database_path=args.pdb_seqres_database_path,
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

            print("Generating alignments for pdb_id/sequence: %s:%s" % (tag,seq))
            alignment_runner.run(
                tmp_fasta_path, local_alignment_dir
            )
        else:
            #copy existing alignment_dir
            print("Copying directory %s to destination %s" % (seq_alignment_dir_dict[seq], local_alignment_dir)) 
            shutil.copytree(seq_alignment_dir_dict[seq], local_alignment_dir, dirs_exist_ok=True)
            
            
        os.remove(tmp_fasta_path)


def generate_features(fasta_path, alignment_dir, seq_list, multimer_pdb_id_list, args):

    print('GENERATING features.pkl for multimer')

    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=args.template_mmcif_dir,
        max_template_date=args.max_template_date,
        max_hits=4,
        kalign_binary_path=args.kalign_binary_path,
        release_dates_path=args.release_dates_path,
        obsolete_pdbs_path=args.obsolete_pdbs_path
    )
    data_processor_monomer = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )
    data_processor_multimer = data_pipeline.DataPipelineMultimer(
        monomer_data_pipeline=data_processor_monomer
    )
    feature_dict = data_processor_multimer.process_fasta(
        fasta_path=fasta_path, alignment_dir=alignment_dir,
    )
    features_output_path = os.path.join(alignment_dir, 'features.pkl')
    with open(features_output_path, 'wb') as f:
        pickle.dump(feature_dict, f, protocol=4)
    print('SAVED %s' % features_output_path)

    print('GENERATING features.pkl for monomer')

    for i in range(0,len(seq_list)):
        chain_alignment_dir = '%s/%s' % (alignment_dir, multimer_pdb_id_list[i])
        fasta_path = write_fasta_file([seq_list[i]], [multimer_pdb_id_list[i]], chain_alignment_dir, multimer_pdb_id_list[i])
        feature_dict = data_processor_monomer.process_fasta(
            fasta_path=fasta_path, alignment_dir=chain_alignment_dir,
        )
        features_output_path = os.path.join(chain_alignment_dir, 'features.pkl')
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
        "--uniprot_id_list", type=list_of_strings, default=None,
        help="",
    )
    parser.add_argument(
        "--pdb_id_list", type=list_of_strings, default=None,
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
        #assumes fasta_path is formatted as protein1-protein2_A, protein1-protein2_B
        #assumes fasta_path is named as uniprot1-uniprot2.fasta
        with open(args.fasta_path) as fasta_file:
            seqs, tags = parsers.parse_fasta(fasta_file.read())
        
        multimer_pdb_wo_chain_id_str = tags[0].split('_')[0]
        print("multimer_pdb_wo_chain_id_str: %s" % multimer_pdb_wo_chain_id_str)
        print("Generating alignments for sequences:" )
        print(seqs)
        multimer_uniprot_str = args.fasta_path.split('/')[-1].split('.')[0]
        msa_dst_dir = '%s/%s/%s' % (args.msa_save_dir, multimer_uniprot_str, multimer_pdb_wo_chain_id_str) #subdirectories are inferred from fasta file 
        print("Saving alignments in %s" % msa_dst_dir)
        os.makedirs(msa_dst_dir, exist_ok=True)
       
        fasta_dst_path = '%s/%s.fasta' % (msa_dst_dir, multimer_uniprot_str) 
        shutil.copyfile(args.fasta_path, fasta_dst_path)

        precompute_alignments(args.fasta_path, msa_dst_dir, args)
        generate_features(args.fasta_path, msa_dst_dir, seqs, tags, args)
        sys.exit() 

     
    uniprot_id_list = args.uniprot_id_list
    pdb_id_list = args.pdb_id_list

    if uniprot_id_list is None and pdb_id_list is None:
        raise ValueError("Both uniprot_id_list and pdb_id_list cannot be none")
    elif uniprot_id_list is not None and pdb_id_list is not None:
        raise ValueError("Both uniprot_id_list and pdb_id_list cannot be passed in")
    
    if pdb_id_list is not None:
        for pdb in pdb_id_list:
            if len(pdb.split('_')) != 2:
                raise ValueError("Every entry in the pdb_id_list must be of format XXXX_Y, where XXXX is the pdb_id and Y is the chain_id")

    pdb_openprotein_dict = {} 

    if pdb_id_list is None:
        #populate corresponding pdbs for each uniprot.  
        pdb_id_list = [] 
        uniprot_pdb_df = pd.read_csv('%s/uniprot_pdb.csv' % script_dir)
        uniprot_pdb_df = uniprot_pdb_df.iloc[1:,]
        uniprot_pdb_df.columns = ['pdb_list']
        uniprot_pdb_df['uniprot_id'] = uniprot_pdb_df.index
        uniprot_pdb_df = uniprot_pdb_df.reset_index(drop=True)
        uniprot_pdb_df = uniprot_pdb_df[['uniprot_id','pdb_list']]

        uniprot_pdb_dict = {} #mapping uniprot_id to all pdb_ids
        for uniprot_id in uniprot_id_list: 
            pdb_list = list(uniprot_pdb_df[uniprot_pdb_df['uniprot_id'] == uniprot_id]['pdb_list'])[0].split(';')
            uniprot_pdb_dict[uniprot_id] = pdb_list

        print('ALL PDBs for each unique Uniprot ID')
        print(uniprot_pdb_dict)

        openprotein_pdb_dict = {}
        uniprot_pdb_max_seq_len_dict = {}

        for i,uniprot_id in enumerate(uniprot_id_list):
            openprotein_pdb_dict[uniprot_id] = [] #mapping uniprot_id to pdb_ids in openprotein 
            if len(uniprot_pdb_dict[uniprot_id]) == 0 or not(args.use_openprotein_alignments):
                rel_pdb_id_w_chain = 'protein%d' % i
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
                    print('GETTING PDB IN OPENPROTEIN WITH MAXIMUM LENGTH')
                    print(openprotein_pdb_dict) 
                    if uniprot_id not in uniprot_pdb_max_seq_len_dict:
                        #this pdb_id corresponds to pdbs corresponding to uniprot_id present 
                        #in openprotein_pdb_dict with the longest sequence 
                        rel_pdb_id_w_chain = get_pdb_w_max_seq_len(openprotein_pdb_dict[uniprot_id], uniprot_id)
                        uniprot_pdb_max_seq_len_dict[uniprot_id] = rel_pdb_id_w_chain
                    else: #this uniprot_id already exists in uniprot_pdb_max_seq_len_dict (i.e homodimer, etc.)
                        rel_pdb_id_w_chain = uniprot_pdb_max_seq_len_dict[uniprot_id]
                else:
                    rel_pdb_id_w_chain = 'protein%d' % i
        
            pdb_id_list.append(rel_pdb_id_w_chain)

    elif uniprot_id_list is None:
        uniprot_id_list = []
        for pdb_id in pdb_id_list:
            uniprot_id = get_uniprot_id(pdb_id)
            uniprot_id_list.append(uniprot_id) 
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



    print("pdb_id list:")
    print(pdb_id_list)
    print('************')
    print("uniprot_id_list")
    print(uniprot_id_list)
    print('************')
    
    multimer_uniprot_str = '-'.join(uniprot_id_list)       

    pdb_id_wo_chain_list = [p.split('_')[0] for p in pdb_id_list] 
    multimer_pdb_wo_chain_id_str = '-'.join(pdb_id_wo_chain_list)

    seq_list = [] 
    multimer_pdb_id_list = []
    multimer_chain_id_list = []
    multimer_chain_info_dict = {}
           
    for i,(pdb_id,uniprot_id) in enumerate(zip(pdb_id_list,uniprot_id_list)):

        multimer_chain_id = num_to_chain(i)
        multimer_pdb_id = '%s_%s' % (multimer_pdb_wo_chain_id_str, multimer_chain_id)

        multimer_pdb_id_list.append(multimer_pdb_id)
        multimer_chain_id_list.append(multimer_chain_id)

        chain_alignment_dir = '%s/%s/%s/%s' % (args.msa_save_dir, multimer_uniprot_str, multimer_pdb_wo_chain_id_str, multimer_pdb_id)
        Path(chain_alignment_dir).mkdir(parents=True, exist_ok=True)

        if pdb_id not in pdb_openprotein_dict:
            seq = get_pdb_id_seq(pdb_id, uniprot_id) #use sequence corresponding to pdb_id
            seq_list.append(seq)
        else:
            prefix = 'pdb/%s' % pdb_id
            objs = list(bucket.objects.filter(Prefix=prefix))
            
            bfd_src_path = ''
            mgnify_src_path = '' 
            uniref90_src_path = '' 
            for obj in objs:
                output_fname = '%s/%s/%s' % (args.msa_save_dir, multimer_uniprot_str, obj.key)
                obj_path = os.path.dirname(output_fname)
                Path(obj_path).mkdir(parents=True, exist_ok=True)
                bucket.download_file(obj.key, output_fname)
                if 'bfd' in obj.key:
                    bfd_src_path = output_fname
                elif 'mgnify' in obj.key:
                    mgnify_src_path = output_fname
                elif 'uniref90' in obj.key:
                    uniref90_src_path = output_fname

            bfd_dst_path = '%s/bfd_uniref_hits.a3m' % chain_alignment_dir 
            mgnify_dst_path = '%s/mgnify_hits.a3m' % chain_alignment_dir
            uniref90_dst_path = '%s/uniref90_hits.a3m' % chain_alignment_dir

            if os.path.exists(mgnify_src_path):
                shutil.copyfile(mgnify_src_path, mgnify_dst_path)
            if os.path.exists(uniref90_src_path):
                shutil.copyfile(uniref90_src_path, uniref90_dst_path) 
            if os.path.exists(bfd_src_path):
                shutil.copyfile(bfd_src_path, bfd_dst_path)
         
            '''convert a3m to sto
               reformat .sto so that it is compatible with openfold pipeline and save in appropriate directory 
            print('converting a3m to sto') 
            if os.path.exists(mgnify_tmp_path):
                subprocess.run(['perl', perl_script, mgnify_tmp_path, mgnify_dst_path])
                format_sto(mgnify_dst_path)
                os.remove(mgnify_tmp_path)
            if os.path.exists(uniref90_tmp_path):
                subprocess.run(['perl', perl_script, uniref90_tmp_path, uniref90_dst_path])
                format_sto(uniref90_dst_path)
                os.remove(uniref90_tmp_path)'''  

            shutil.rmtree('%s/%s/%s' % (args.msa_save_dir, multimer_uniprot_str, 'pdb'))

            #this is the sequence for which the msa was generated 
            if os.path.exists(bfd_src_path):
                msa_seq = get_query_seq_from_msa(bfd_dst_path)
            elif os.path.exists(uniref90_dst_path):
                msa_seq = get_query_seq_from_msa(uniref90_dst_path)

            seq_list.append(msa_seq)

        multimer_chain_info_dict[multimer_pdb_id] = {}
        multimer_chain_info_dict[multimer_pdb_id]['pdb_id'] = pdb_id
        multimer_chain_info_dict[multimer_pdb_id]['uniprot_id'] = uniprot_id
        multimer_chain_info_dict[multimer_pdb_id]['seq'] = seq_list[-1]        

    msa_dst_dir = '%s/%s/%s' % (args.msa_save_dir, multimer_uniprot_str, multimer_pdb_wo_chain_id_str) #subdirectories are inferred from fasta file 
    multimer_chain_info_output_path = os.path.join(msa_dst_dir, 'multimer_chain_info.json')
    with open(multimer_chain_info_output_path, 'w') as f:
        json.dump(multimer_chain_info_dict, f)
    print('SAVED %s' % multimer_chain_info_output_path)

    fasta_path = write_fasta_file(seq_list, multimer_pdb_id_list, msa_dst_dir, multimer_uniprot_str)

    #compute remaining alignments -- this includes pdb_hits from hhmsearch  
    print('COMPUTING ALIGNMENTS')
    precompute_alignments(fasta_path, msa_dst_dir, args)
    print('ALIGNMENTS COMPTUED')

    generate_features(fasta_path, msa_dst_dir, seq_list, multimer_pdb_id_list, args)
