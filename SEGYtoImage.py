import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from pathlib import Path

from obspy import read

#from scipy.ndimage import gaussian_filter

import os 
import sys

def stream_to_numpy(st,normalise=True):
    # Creates a numpy array for geophone stream
    fs=st[0].stats.sampling_rate
    data_length=st[0].stats.npts
    out_data=np.zeros((len(st),data_length))
    
    # Integrate to convert to displacement from velocity
    #     st=st.integrate()
    
    for i in range(len(st)):
        in_data=st[i].data
        if normalise==True:
            # Normalise the data
            tmp=in_data/np.max(in_data)
            out_data[i]=tmp-np.mean(tmp)
        elif normalise==False:
            out_data[i]=in_data-np.mean(in_data)
        else:
            print("Would you like to normalise the data? True of False?")
            break
    return out_data, fs
    
    
def fk_filter(data, fs, ch_space, max_wavenum, min_wavenum, max_freq, min_freq, plot=True):
    """FK filter for a 2D DAS numpy array. Returns a filtered image."""
    
    # Detrend by removing the mean 
    data=data-np.mean(data)
    
    # Apply a 2D fft transform
    fftdata=np.fft.fftshift(np.fft.fft2(data.T))
    
    freqs=np.fft.fftfreq(fftdata.shape[1],d=(1./fs))
    wavenums=np.fft.fftfreq(fftdata.shape[0],d=ch_space)

    freqs=np.fft.fftshift(freqs) 
    wavenums=np.fft.fftshift(wavenums)

    freqsgrid=np.broadcast_to(freqs,fftdata.shape)   
    wavenumsgrid=np.broadcast_to(wavenums,fftdata.T.shape).T
    
    # Define mask and blur the edges 
    mask=np.logical_and(np.logical_and(np.logical_and(\
        abs(wavenumsgrid)<=max_wavenum,\
        abs(wavenumsgrid)>min_wavenum),\
        abs(freqsgrid)<max_freq),\
        abs(freqsgrid)>min_freq)
    x=mask*1.
    blurred_mask = ndimage.gaussian_filter(x, sigma=3)
    
    # Apply the mask to the data
    ftimagep = fftdata * blurred_mask
    ftimagep = np.fft.ifftshift(ftimagep)
    
    # Finally, take the inverse transform and show the blurred image
    imagep = np.fft.ifft2(ftimagep)

    imagep = imagep.real
    
    if plot==True:
        # Plots the filter, with area remove greyed out
        plt.figure(figsize=[6,6])
        img1 = plt.imshow(np.log10(abs(fftdata)), interpolation='bilinear',extent=[-fs/2,fs/2,-1/(2*ch_space),1/(2*ch_space)],aspect='auto')
        img1.set_clim(-5,5)
        img1 = plt.imshow(abs(blurred_mask-1),cmap='Greys',extent=[-fs/2,fs/2,-1/(2*ch_space),1/(2*ch_space)],alpha=0.2,aspect='auto')
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Wavenumber (1/m)')
        #plt.xlim(-200,200)
        #plt.ylim(-0.3,0.3)
        plt.show()
        
    return imagep



if __name__ == "__main__":

    directory = 'C:/Projects/FORGE/SEGY/2024/097/'
    SEGYdir = 'SEGY-Class2'
    
    #glob_path       = Path(r"Z:\SeiBer\ManualEventExtraction\SEGY")
    glob_path       = Path(directory)
    #glob_path       = Path(r"Z:\SeiBer\ManualEventExtraction\SEGY_2")
    EvtData         = [str(pp) for pp in glob_path.glob("*.sgy")]
    EvtData.sort()
    
    if not os.path.exists(directory+SEGYdir):
        os.makedirs(directory+SEGYdir)


    #print (EvtData)
    #noiseDir        = Path(r"G:\LKAB2_MSeis\01_Detect\Full_04\LKAB2_full04_Noise")
    #evtDir          = Path(r"G:\LKAB2_MSeis\01_Detect\Full_04\LKAB2_full04_SortedEvt")
    
        
        
    #first_ch   = 1000
    #last_ch    = 1900
    first_ch   = 1
    last_ch    = 2486

    
    #fs              = 2000.
    ch_space        =  1.0
    
    #bandpass filter
    freqmin = 10
    freqmax = 100


    # f-k filter
    max_wavenum     =   0.1
    min_wavenum     =   0.01   
    max_freq        =   200
    min_freq        =   1
    
   
    
    for file in EvtData:
            
            first_samp = np.random.randint(0,900)
            last_samp = first_samp+2000
            #if int(file[-12:-7]) > 0:
            st =read(file,format='SEGY')
            #st.detrend("linear")
            #st.taper(max_percentage=0.01, type="hann")
            #st.filter("bandpass", freqmin=10, freqmax=50, zerophase=True)
            st.filter("bandpass", freqmin=freqmin, freqmax=freqmax, zerophase=True)
            #print(st[0].stats.delta)
            
            data, fs    =   stream_to_numpy(st, normalise=False)  
            print(file)
            data = data[first_ch:last_ch,first_samp:last_samp]

            vMine = np.percentile(data,10.0)
            vMaxe = np.percentile(data,90.0)
            
            plt.figure(figsize=(8,6))
            
            #plt.imshow(imagep, aspect='auto', vmin=-800,vmax=800, cmap="gray")
            #plt.imshow(imagep, aspect='auto', vmin=-200,vmax=200, cmap="gray", extent=(0,300,0,1308))
            plt.imshow(data, aspect='auto', vmin=-abs(vMine),vmax=abs(vMaxe), cmap="gray")
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0, 0)
            #plt.xlabel('Time sample\n',fontsize=16)
            #plt.ylabel('Channel\n',fontsize=16)
            #plt.yticks(fontsize=14)
            #plt.xticks(fontsize=14)
            plt.axis('off')
            #plt.tight_layout()
            plt.tight_layout()
            outfile = 'ExampleEvt_'+file[-12:-4]+'.png'
            plt.savefig(directory+SEGYdir+'/'+outfile, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
            
            

