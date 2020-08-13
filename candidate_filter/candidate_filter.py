import optparse
import argparse
import json
import os
import reading_cands
import cluster_cands
import spatial_rfi
import filtering
import pandas as pd
import numpy as np
import known_filter
import time


def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Command line arguments for the candidate filtering.')
    parser.add_argument('-i', '--input', type=str, default='', metavar=('input_files'),
                        help="Path to the input files.", nargs='+')
    parser.add_argument('-o', '--output', type=str, default='', metavar=('output_path'),
                        help="Base name of the output csv files")
    default_config_path = f"{os.path.dirname(__file__)}/default_config.json"
    parser.add_argument('-c', '--config', type=str, default=default_config_path,
                        metavar=('config_file'), help="Path to config file.")
    parser.add_argument('-p', '--plot', action='store_true',
                        help="Plot diagnostic plots of the clusters.")
    parser.add_argument('-H','--harmonics',type=int,help='harmonic number to search upto',default=16)
    parser.add_argument('--p_tol',type=float,help='period tolerance',dest="p_tol",default=5e-4)
    parser.add_argument('--dm_tol',type=float,help='dm tolerance',dest="dm_tol",default=5e-3)
    parser.add_argument('--par',type=str,help='Path to par files of known pulsars',dest="par_path",default='/beegfs/u/prajwalvp/presto_ephemerides/Ter5/par_files_scott')
    parser.add_argument('--rfi',type=str,help='known birdie list name',dest="birdies",default='known_rfi.txt')
    args = parser.parse_args()
    return args









def main(args):
    # Main function

    # Load config
    with open(args.config) as json_data_file:
        config = json.load(json_data_file)

    # Read files into a single pandas DataFrame
    df_cands_ini, obs_meta_data = reading_cands.read_candidate_files(
        args.input)


    # Write out main csv
    df_cands_ini.to_csv('all_cands.csv')

    # Get candidate periods and dms
    cand_periods = df_cands_ini['period'].to_numpy()
    cand_freqs = 1/cand_periods
    cand_dms = df_cands_ini['dm'].to_numpy()
    cand_snrs = df_cands_ini['snr'].to_numpy()

    # Label known RFI sources
    known_rfi_indices = known_filter.get_known_rfi(cand_freqs,args)
    print("Number of RFI instances: %d"%len(known_rfi_indices))
    time.sleep(5)

    # Get known pulsar periods and dms
    known_psrs = known_filter.get_params_from_pars(args.par_path)
    known_freqs = np.array(known_psrs['F0'],dtype=float)
    known_dms = np.array(known_psrs['DM'],dtype=float)

    # Label known pulsar sources from input par files
    known_psr_indices,known_ph_indices = known_filter.get_known_psr(args,known_psrs,known_freqs,known_dms,cand_freqs,cand_dms,cand_snrs) 


    # Retain candidates which are not known rfi or pulsars
    all_known_indices =  list(set(list(known_rfi_indices) + known_psr_indices + known_ph_indices))
    print("Number of dropped candidates: %d"%len(all_known_indices))
    df_cands_remain = df_cands_ini.drop(df_cands_ini.index[all_known_indices]) 
    df_cands_remain.reset_index(drop=True)
    print (df_cands_ini.columns)
    print (df_cands_remain.columns)

    df_cands_remain.to_csv('remaining_cands.csv')

    # Create clusters from remaining candidates
    #df_cands_clustered = cluster_cands.cluster_cand_df(
    #    df_cands_ini, obs_meta_data, config)
    df_cands_clustered = cluster_cands.cluster_cand_df(
        df_cands_remain, obs_meta_data, config)

    # Find spatial RFI and write out details about clusters
    df_clusters = spatial_rfi.label_spatial_rfi(df_cands_clustered, config)


    # Label bad clusters
    df_cands_filtered, df_clusters_filtered = filtering.filter_clusters(df_cands_clustered,
                                                                        df_clusters, config)

    # Write out candidate list
    df_cands_filtered.to_csv(f"{args.output}_cands.csv")
    # Write out cluster list
    df_clusters_filtered.to_csv(f"{args.output}_clusters.csv")

    # Write out candidate lists for single beams
    output_folder = f"{os.path.dirname(args.output)}/single_beams/"
    try:
        os.mkdir(output_folder)
    except FileExistsError:
        pass
    unique_file_idxs = df_cands_filtered['file_index']
    for file_index in unique_file_idxs:
        df_file = df_cands_filtered[df_cands_filtered['file_index'] == file_index]
        file_name = os.path.basename(os.path.dirname(df_file['file'].iloc[0]))
        df_file.to_csv(f"{output_folder}{file_name}.csv")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
