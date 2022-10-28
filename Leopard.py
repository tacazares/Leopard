import os
import numpy as np
import glob
import argparse
from subprocess import Popen
from os import path, getcwd, makedirs, error, walk, environ
import pyBigWig
import sys
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description="run Leopard predictions")
    parser.add_argument('-tf', '--transcription_factor', default='HNF4A', type=str,
        help='transcript factor')
    parser.add_argument('-te', '--test', default='liver', type=str,
        help='test cell type')
    parser.add_argument('-chr', '--chromosome', default='chr21', nargs='+', type=str,
        help='test chromosome')
    parser.add_argument('-i', '--input_bigwig', type=str, required=True,
        help='Input bigwig file')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
        help='Output Directory')
    parser.add_argument('-m', '--mode', default='fast', type=str,
        help='prediction mode (complete or fast)')
    parser.add_argument('-reso', '--resolution', default='1bp', type=str,
        help='resolution of prediction')
    parser.add_argument('-n', '--name',required=True, type=str,
        help='outputfilename with extension')
    args = parser.parse_args()
    return args

def write_predictions_to_bigwig(df,
                                output_filename,
                                ):
    """Write the predictions dataframe into a bigwig file
    Args:
        df (pd.DataFrame): The dataframe of BED regions with prediction scores
        output_filename (str): The output bigwig filename
        chrom_sizes_dictionary (dict): A dictionary of chromosome sizes used to form the bigwig file
        chromosomes (list): A list of chromosomes that you are predicting in
        agg_mean (bool, optional): use aggregation method of mean. Defaults to True.
    Returns:
        object: Writes a bigwig file
        
    Example:
    """
    print("Writing file")
    with dump_bigwig(output_filename) as data_stream:
        print("Adding header")
        # Make the bigwig header using the chrom sizes dictionary
        header = [("chr1", 249250621)]

        # Add header to bigwig
        data_stream.addHeader(header)

        # Bigwig files need sorted intervals as input
        #tmp_chrom_df = df.sort_values(by=["chrom", "start"])
        
        print("Writing values")
        # Write all entries for the chromosome
        data_stream.addEntries(chroms=df["chrom"].tolist(),
                                   starts=df["start"].tolist(),
                                   ends=df["end"].tolist(),
                                   values=df["score"].tolist()
                                   )

def get_absolute_path(p, cwd_abs_path=None):
    cwd_abs_path = getcwd() if cwd_abs_path is None else cwd_abs_path
    return p if path.isabs(p) else path.normpath(path.join(cwd_abs_path, p))

def dump_bigwig(location):
    """Write a bigwig file to the location
    Args:
        location (str): The path to desired file location
    Returns:
        bigwig stream: An opened bigwig for writing
    """
    return pyBigWig.open(get_absolute_path(location), "w")

def main():
    args = get_args()

    the_tf = args.transcription_factor
    the_test = args.test
    chr_all = args.chromosome
#    print(chr_all)
    if not isinstance(args.chromosome, list):
        chr_all = [chr_all] # if one chr, convert it into list
#    print(chr_all)
    the_reso = args.resolution

    # list existing models
    the_path = 'code_' + the_reso + '/' + the_tf + '/'
    models = glob.glob(the_path + 'weights_*1.h5')
    print(models)
    _,the_train,the_vali,_ = models[0].split('/')[-1].split('_')
    num_par = len(glob.glob(the_path + 'weights_' + the_train + '_' + the_vali + '_*.h5'))

    if args.mode!='complete':
        num_par = 1
        print('fast mode - run ' + str(num_par) + ' replicate')
    else:
        print('complete mode - run ' + str(num_par) + ' replicates trained on different seeds')

    # run prediction in parallel
    for i in np.arange(1,num_par+1):
        print('run replicate ' + str(i) + '/' + str(num_par) + ':')
        cmd_all=[]
        for j in np.arange(len(models)):
            _,the_train,the_vali,_ = models[j].split('/')[-1].split('_')
            the_model = the_path + 'weights_' + the_train + '_' + the_vali + '_' + str(i) + '.h5'
            print('model: ' + the_train + '_' + the_vali)
            cmd = ['python', 'code_' + the_reso + '/predict.py', '-m', the_model, \
                '-tf',  the_tf, '--input_bigwig', args.input_bigwig, '-o', args.output_dir, '-te', the_test, '-chr'] + chr_all + \
                ['-para', str(len(models))]
            cmd_all.append(cmd)
        procs = [ Popen(i) for i in cmd_all ]
        for p in procs: # run in parallel
            p.wait()

    # stacking prediction from multiple models
    print('combining predictions from different models')
    for the_chr in chr_all:
        print(the_chr)
        combo_name = os.path.join(args.output_dir, 'pred_' + the_tf + '_' + the_test + '_' + the_chr)
        for i in np.arange(1,num_par + 1):
            for j in np.arange(len(models)):
                _,the_train,the_vali,_ = models[j].split('/')[-1].split('_')
                if i==1 and j==0:
                    pred = np.load(combo_name + '_weights_' + the_train + '_' + the_vali + '_' + str(i) + '.npy')
                else:
                    pred += np.load(combo_name + '_weights_' + the_train + '_' + the_vali + '_' + str(i) + '.npy')
        pred = pred / float(len(models)) / float(num_par)
        np.save(combo_name, pred)
        
        print ("Creating a dataframe")
        # Create a dataframe from the NP array
        preds_DF = pd.DataFrame(pred)
        
        print("Create a col with chr ID")
        # add a column with the chromosome number, in this case we are only using chr1
        preds_DF["chrom"] = "chr1"

        print("Create a col with start")
        # add a column with the start position based on the index
        preds_DF["start"] = preds_DF[0].index

        print("Create stop")
        # add a column with the stop position 1 bp from the start
        preds_DF["end"] = preds_DF[0].index + 1

        print("Rename cols")
        # rename the columns
        preds_DF.columns = ["score", "chrom", "start", "end"]

        # FIll na values with a 0; might need to change later to be more accurate
        preds_DF.fillna(0, inplace=True)

        # write to csv file. 
        #preds_DF[["chrom", "start", "end", "score"]].to_csv(basename_file + "_1bp.bed.gz", sep="\t", index=False, header=False)

        output_bigwig_name = os.path.join(args.output_dir, args.name)
        write_predictions_to_bigwig(preds_DF, args.name)
        
        print("Cleanup")
        # remove individual predictions from each model
        for i in np.arange(1,num_par + 1):
            for j in np.arange(len(models)):
                _,the_train,the_vali,_ = models[j].split('/')[-1].split('_')
                os.system('rm ' + combo_name + '_weights_' + the_train + '_' + the_vali + '_' + str(i) + '.npy')


if __name__ == '__main__':
    main()



