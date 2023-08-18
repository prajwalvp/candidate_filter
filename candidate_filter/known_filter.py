import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
import xml.etree.ElementTree as ET
import os
import optparse
import re
import subprocess
import itertools
import logging
import pandas as pd



log = logging.getLogger('filter_and_presto_fold')
FORMAT = "[%(levelname)s - %(asctime)s - %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)

log.setLevel('INFO')



def get_known_rfi(cand_freqs, birdie_list_file, harmonics, p_tol):
    # Get all candidate frequencies
    known_rfi_freqs = np.loadtxt(birdie_list_file, dtype=float) 
    h = np.arange(1, harmonics + 1)
    ratios = np.outer(h, 1.0 / h)
    all_ratios = np.unique(np.sort(ratios.ravel()))

    w_harmonics=[]
    for number in all_ratios:
        for rfi in known_rfi_freqs:
            subs = []
            for a in cand_freqs:
                if (abs(a - number * rfi) / a) < p_tol:
                    print("Matched candidate {} Hz with harmonic {} of RFI {} Hz".format(a, number, rfi))
                    subs.append(a)
            w_harmonics.append(subs)    
    print(w_harmonics)
    all_harmonics=sorted(list(set(list(itertools.chain(*w_harmonics)))))
    print("Matched the following harmonics: {}".format(all_harmonics))
    rfi_indices = []
    for h in all_harmonics:
        rfi_indices.extend(np.where(cand_freqs==h)[0])
    log.info("Number of known RFI based candidates: {}".format(len(rfi_indices)))    
    return np.array(rfi_indices)



def a_to_pdot(P_s, acc_ms2):
    LIGHT_SPEED = 2.99792458e8                 # Speed of Light in SI
    return P_s * acc_ms2 /LIGHT_SPEED


def period_modified(p0,pdot,no_of_samples,tsamp,fft_size):
    if (fft_size==0.0):
        return p0 - pdot*float(1<<(no_of_samples.bit_length()-1))*tsamp/2
    else:
        return p0 - pdot*float(fft_size)*tsamp/2

def sortKeyFunc(s):
    return int(re.search('f1_(.*)_',s).group(1).split('_')[0])


def get_params_from_pars(par_path):

    known_psrs={}
    known_psrs['names']=[]
    known_psrs['F0']=[]
    known_psrs['DM']=[]

    known_pars=glob.glob(par_path+'/*.par')
    for kp in known_pars:
        known_psrs['names'].append(os.path.basename(kp)[:-4])
        known_psrs['F0'].append(subprocess.getoutput('grep \"F0\" %s'%kp).split()[1])
        known_psrs['DM'].append(subprocess.getoutput('grep \"DM\" %s'%kp).split()[1])

    return known_psrs


def get_params_from_xml(xml_path):
    xml_file = xml_path + '/' + 'overview.xml'
    tree = ET.parse(xml_file)
    root = tree.getroot()
    cand_periods = np.array([a.text for a in root.findall("candidates/candidate/period")],dtype=float)
    cand_accs = np.array([a.text for a in root.findall("candidates/candidate/acc")],dtype=float)
    tsamp = float(root.find("header_parameters/tsamp").text)
    fft_size = float(root.find('search_parameters/size').text)
    nsamples = int(root.find("header_parameters/nsamples").text)


    mod_periods=[]
    pdots= []
    for i in range(len(cand_periods)):
        Pdot = a_to_pdot(cand_periods[i],cand_accs[i])
        mod_period.append(period_modified(cand_periods[i],Pdot,xml['no_of_samples'],xml['tsamp'],xml['fft_size']))
        pdot.append(Pdot) 

    cand_mod_periods = np.asarray(mod_period,dtype=float)

    # Modify periods
    cand_freqs = 1/cand_mod_periods
    cand_dms = np.array([a.text for a in root.findall("candidates/candidate/dm")],dtype=float)
    cand_snrs = np.array([a.text for a in root.findall("candidates/candidate/snr")],dtype=float)
    cand_nh = np.array([a.text for a in root.findall("candidates/candidate/nh")],dtype=int)

    return cand_mod_periods,cand_freqs,cand_dms,cand_snrs,cand_nh


def get_known_psr(args,known_psrs,known_freqs,known_dms,cand_freqs,cand_dms,cand_snrs):
    psr_indices=[]
    reduced_ph_indices=[]

    #Pulsar indices
    known_freq_indices,cand_freq_indices = np.where(np.isclose(*np.ix_(known_freqs, cand_freqs), rtol=args.p_tol))

    with open('possible_known_psrs.txt','a') as f:
        #f.truncate(0)
        #f.write(xml_path+'\n')
        #f.write("PULSAR CAND_NO SNR F0 CAND_F0 DM CAND_DM\n")
        for ind in sorted(cand_freq_indices):
           idx = known_freq_indices[np.where(cand_freq_indices==ind)[0][0]]
           if np.isclose(cand_dms[ind],known_dms[idx],rtol=args.dm_tol):
               f.write(known_psrs['names'][idx]+' '+str(ind)+' '+str(cand_snrs[ind])+' '+str(known_freqs[idx])+' '+str(cand_freqs[ind])+' '+str(known_dms[idx])+' '+str(cand_dms[ind])+'\n')
               log.info("Potential redetection....")
               log.info("PULSAR CAND_NO SNR F0 CAND_F0 DM CAND_DM")
               log.info(known_psrs['names'][idx]+' '+str(ind)+' '+str(cand_snrs[ind])+' '+str(known_freqs[idx])+' '+str(cand_freqs[ind])+' '+str(known_dms[idx])+' '+str(cand_dms[ind])+'\n')
               psr_indices.append(ind)     
        f.close()

    #Harmonic indices - Reduce to harmonics of Ter5A and C only
    h = np.arange(1,args.harmonics)
    ratios = np.outer(h,1.0/h)
    all_ratios = np.unique(np.sort(ratios.ravel()))

    known_freqs = [86.48163691]
    for i,freq in enumerate(known_freqs):
        r = all_ratios*freq            
        p = abs(1-cand_freqs/r[np.searchsorted(r,cand_freqs)-1])

        if np.searchsorted(r,cand_freqs).max()==len(r):
            tmp = np.searchsorted(r,cand_freqs)    
            fixed = np.where(tmp==tmp.max(), tmp.max()-1, tmp) 
            q = abs(1-cand_freqs/r[fixed])
        else:
            q = abs(1-cand_freqs/r[np.searchsorted(r,cand_freqs)])
        l = np.vstack((p, q)).min(axis=0)
        if '86.48' in str(freq):    
            ph_indices = np.where(l<5e-4)[0] # relax the tolerance for Ter5A by order of mag.
        else:
            ph_indices = np.where(l<args.p_tol)[0]

        for ph in sorted(ph_indices):
            if '86.48' in str(freq):
                if np.isclose(cand_dms[ph],known_dms[i],rtol=5e-3): # relax dm tolerance for Ter5A
                    reduced_ph_indices.append(ph)
            else:
                if np.isclose(cand_dms[ph],known_dms[i],rtol=args.dm_tol):
                    reduced_ph_indices.append(ph)
                #print freq,known_psrs['names'][i],ph                  
    return psr_indices,reduced_ph_indices




if __name__=="__main__":
        
    # Get all xmls 
    
    xml_path = opts.xml_path

    log.info("Reading XML file from %s"%xml_path) 
    tree = ET.parse(xml_path+'/overview.xml')
    root = tree.getroot()
    cand_periods = np.array([a.text for a in root.findall("candidates/candidate/period")],dtype=float)
    cand_dms = np.array([a.text for a in root.findall("candidates/candidate/dm")],dtype=float)
    cand_accs = np.array([a.text for a in root.findall("candidates/candidate/acc")],dtype=float)
    cand_snrs = np.array([a.text for a in root.findall("candidates/candidate/snr")],dtype=float)
    cand_nh = np.array([a.text for a in root.findall("candidates/candidate/nh")],dtype=int)
    tsamp = float(root.find("header_parameters/tsamp").text)
    fft_size = float(root.find('search_parameters/size').text)
    nsamples = int(root.find("header_parameters/nsamples").text)


    log.info("Total number of candidates produced: %d"%len(cand_periods))


    # Modify period
    log.info("Calculating modified period based on starting epoch reference")
    mod_periods=[]
    pdots = []
    for i in range(len(cand_periods)):
        Pdot = a_to_pdot(cand_periods[i],cand_accs[i])
        mod_periods.append(period_modified(cand_periods[i],Pdot,nsamples,tsamp,fft_size))
        pdots.append(Pdot)

    cand_mod_periods = np.asarray(mod_periods,dtype=float)
    cand_freqs = 1/cand_mod_periods



    log.info("Retrieving RFI candidates from catalogue of sources....")
    rfi_indices = get_possible_rfi_indices(opts,cand_freqs)
    log.info("Done. Number of RFI based candidates: %d"%len(rfi_indices))
    print (rfi_indices)
    
    log.info("Retrieving known pulsar parameters from par files...")        
    known_psrs = get_params_from_pars(opts.par_path)
    known_freqs = np.array(known_psrs['F0'],dtype=float)
    known_dms = np.array(known_psrs['DM'],dtype=float)
    log.info("Done")


    log.info("Getting possible pulsar  and corresponding harmonic candidates indices")
    p_indices,ph_indices = get_possible_pulsars_and_harmonics_indices(opts,known_psrs,known_freqs,known_dms,cand_freqs,cand_dms)
    log.info("Done: Number of pulsars: %d , Number of Ter5A and C harmonics: %d"%(len(p_indices),len(ph_indices)))
    print (p_indices,ph_indices)
    


    log.info("Setting up paths for folding") 
    input_name = root.find("search_parameters/infilename").text 
    mask_path = "/beegfs/PROCESSING/TRAPUM/RFIFIND_masks/Ter5_16apr20_frankenmask/frankenbeam_mask_rfifind.mask"
    output_path = xml_path
     


    log.info("First folding the candidates that do not make the RFI,pulsar or harmonics cut")
    remaining_cand_indices = [i for i in range(len(cand_mod_periods)) if i not in rfi_indices and i not in ph_indices and i not in p_indices]
    log.info("Number of caniddates: %d"%(len(remaining_cand_indices)))

