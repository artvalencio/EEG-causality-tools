#===================================================================#
#EEG Causality Tools
#Author: Arthur Valencio
#Contact: arthur_valencio(at)physics.org
#Institution: Institute of Computing, Unicamp; CEPID NeuroMat
#Version: 25-Feb-2022
#Licensing: Open, attribution-only (CC-BY)
#Description: GUI for EEG analysis using correlation or tools
#             from information theory
#Acknowledgments: FAPESP (S.Paulo Research Foundation)
#                 grants 2018/09900-8 and 2013/07699-0
#===================================================================#
#How to cite this work:
#   Valencio, A. EEG Causality Tools. Python-based graphical user
#     interface. Available at:
#     https://github.com/artvalencio/EEG-Causality-Tools
#===================================================================#
#References (third-party libraries used in this work):
#   Gramfort, A. et al. MEG and EEG data analysis with MNE-Python.
#     Frontiers in Neuroscience, 7(267):1–13, 2013.
#     doi:10.3389/fnins.2013.00267.
#   Valencio, A. et al. CaMI-Python: Causality toolbox for Python.
#     2021. Available at: https://github.com/artvalencio/cami-python
#===================================================================#


#libraries
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from PIL import ImageTk, Image
from functools import partial
import os
import shutil
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.signal import find_peaks, peak_widths
from sklearn.metrics.pairwise import euclidean_distances
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import seaborn as sns
from cami import *
import gc
import mne
import mne_connectivity

#global vars
raw1=None
raw2=None
xraw1=None
xraw2=None
events1=None
events2=None
events1_data=None
events2_data=None
event_dict=None
eeg1=None
eeg2=None
epoched_raw1=None
epoched_raw2=None
x1=None
x2=None
montage=None
epochs_delete=None

#load raw data
def load_edf():
    global raw1
    global raw2
    error=0
    try:
        filename=StringVar()
        filename.set(fd.askopenfilename(title="Load raw EEG",filetypes=(('EDF file','*.edf'),('All files','*.*'))))
        raw1=mne.io.read_raw_edf(filename.get()).load_data()
    except:
        showinfo(title="Error",message="Could not load the file")
        error=1
    if error==0:
        window=Toplevel(main)
        window.title("Additional dataset")
        window.geometry("450x200")
        add_edf=IntVar()
        add_edf.set(2)
        Label(window,text="Do you have an additional\nEDF file for a different\ncondition (e.g. DBS ON vs DBS OFF)?",justify=CENTER).grid(row=0,column=0,padx=10,pady=10)
        Radiobutton(window,text="Yes, I have an additional EDF for a different condition",variable=add_edf,value=1).grid(row=1,column=0,padx=10,sticky=W)
        Radiobutton(window,text="No, the different condition is in the same EDF file",variable=add_edf,value=2).grid(row=2,column=0,padx=10,sticky=W)
        Radiobutton(window,text="No, my dataset does not have a different condition to compare", variable=add_edf,value=3).grid(row=3,column=0,padx=10,sticky=W)
        def cont_load():
            global raw1
            global raw2
            if add_edf.get()==1:
                try:
                    filename2=StringVar()
                    filename2.set(fd.askopenfilename(title="Load raw EEG",filetypes=(('EDF file','*.edf'),('All files','*.*'))))
                    raw2=mne.io.read_raw_edf(filename2.get()).load_data()
                except:
                    showinfo(title="Error",message="Could not load the file")
            elif add_edf.get()==2:
                ann_name=StringVar()
                try:
                    optionlist=raw1.annotations.description.tolist()
                    ann_name.set(optionlist[-1])
                except:
                    optionlist=[""]
                    ann_name.set("")
                time_or_ann=IntVar()
                time_or_ann.set(1)
                time_split=StringVar()
                time_split.set("0")
                window2=Toplevel(main)
                window2.title("Split EDF")
                window2.geometry("300x160")
                Label(window2,text="Select moment to split between conditions\n(e.g. the time DBS is switched on)",justify=CENTER).grid(row=0,column=0,columnspan=2,padx=10,pady=10)
                Radiobutton(window2,text="From EDF annotations:",variable=time_or_ann,value=1).grid(row=1,column=0,padx=10,sticky=W)
                OptionMenu(window2,ann_name,optionlist[-1],*optionlist).grid(row=1,column=1,sticky=W)
                Radiobutton(window2,text="From time (seconds):",variable=time_or_ann,value=2).grid(row=2,column=0,padx=10,sticky=W)
                Entry(window2,textvariable=time_split,width=5).grid(row=2,column=1,sticky=W)
                def cont_load2():
                    global raw1
                    global raw2
                    if time_or_ann.get()==1:
                        for i in range(len(raw1.annotations.description)):
                          if raw1.annotations.description[i]==ann_name.get():
                            dbs_time=float(raw1.annotations.onset[i])
                            start1,stop1=raw1.time_as_index([0,dbs_time])
                            start2,stop2=raw1.time_as_index([dbs_time,raw1.times[-1]])
                            raw2=mne.io.RawArray(raw1.get_data(start=start2),raw1.info)
                            raw1=mne.io.RawArray(raw1.get_data(stop=stop1),raw1.info)
                    elif time_or_ann.get()==2:
                        start1,stop1=raw1.time_as_index([0,float(time_split.get())])
                        start2,stop2=raw1.time_as_index([float(time_split.get()),raw1.times[-1]])
                        raw2=mne.io.RawArray(raw1.get_data(start=start2),raw1.info)
                        raw1=mne.io.RawArray(raw1.get_data(stop=stop1),raw1.info)
                    window2.destroy()
                    showinfo(title="File loaded", message="Raw EEG file loaded")
                Button(window2,text="OK",command=cont_load2).grid(row=3,column=0,columnspan=2)
            elif add_edf.get()==3:
                pass
            #if montage is not None: do stuff
            window.destroy()
        Button(window,text="OK",command=cont_load).grid(row=4,column=0,padx=10)
    

#load preprocessed data
def load_fif():
    global eeg1
    global eeg2
    global event_dict
    error=0
    try:
        filename=StringVar()
        filename.set(fd.askopenfilename(title="Load preprocessed EEG",filetypes=(('FIF file','*.fif'),('All files','*.*'))))
        eeg1=mne.read_epochs(filename.get(),preload=True,verbose='ERROR')
        event_dict=eeg1.event_id
    except:
        showinfo(title="Error",message="Could not load the file")
        error=1
    if error==0:
        window=Toplevel(main)
        window.title("Additional dataset")
        window.geometry("450x170")
        add_fif=IntVar()
        add_fif.set(1)
        Label(window,text="Do you have an additional\nFIF file for a different\ncondition (e.g. DBS ON vs DBS OFF)?",justify=CENTER).grid(row=0,column=0,padx=10,pady=10)
        Radiobutton(window,text="Yes, I have an additional FIF for a different condition",variable=add_fif,value=1).grid(row=1,column=0,padx=10,sticky=W)
        Radiobutton(window,text="No, my dataset does not have a different condition to compare", variable=add_fif,value=2).grid(row=3,column=0,padx=10,sticky=W)
        def cont_load():
            global eeg1
            global eeg2
            if add_fif.get()==1:
                try:
                    filename2=StringVar()
                    filename2.set(fd.askopenfilename(title="Load preprocessed EEG",filetypes=(('FIF file','*.fif'),('All files','*.*'))))
                    eeg2=mne.read_epochs(filename2.get(),preload=True,verbose='ERROR')
                except:
                    showinfo(title="Error",message="Could not load the file")
            elif add_fif.get()==2:
                pass
            channel_order_list=list(range(1,len(eeg1.ch_names)+1))
            sel_chan_ord=[]
            for i in range(len(eeg1.ch_names)):
                sel_chan_ord.append(StringVar())
                sel_chan_ord[i].set(str(i+1))
            window2=Toplevel(main)
            eeg1.plot_sensors(show_names=True)
            window2.title("Reorder channels")
            window2.geometry("225x400")
            def onFrameConfigure(canvas):
                canvas.configure(scrollregion=canvas.bbox("all"))
            canvas=Canvas(window2,borderwidth=0,width=250,height=400)
            frame=Frame(canvas)
            scroll=Scrollbar(window2,orient="vertical",command=canvas.yview)
            canvas.configure(yscrollcommand=scroll.set)
            canvas.grid(row=0,column=0,columnspan=4)
            scroll.grid(row=0,column=3,sticky='ns')
            canvas.create_window((15,15), window=frame, anchor="nw")
            frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))
            Label(frame,text="Reorder channels (optional)",justify=CENTER).grid(row=0,column=0,columnspan=2)
            Label(frame,text="Order").grid(row=1,column=0,padx=10,sticky=W)
            Label(frame,text="Electrode channel").grid(row=1,column=1,sticky=E)
            for i in range(len(eeg1.ch_names)):
                channel_order_list2=[channel_order_list[j] for j in range(i,len(channel_order_list))]+[channel_order_list[j] for j in range(i) if i!=0]
                OptionMenu(frame,sel_chan_ord[i],*channel_order_list2).grid(row=i+2,column=0,sticky=W)
                Label(frame,text=eeg1.ch_names[i]).grid(row=i+2,column=1,padx=10,sticky=E)
            def cont_load2():
                global eeg1
                global eeg2
                chan_ord=[int(i.get())-1 for i in sel_chan_ord]
                order=[eeg1.ch_names[i] for i in chan_ord]
                eeg1=eeg1.reorder_channels(order)
                if eeg2 is not None:
                    eeg2=eeg2.reorder_channels(order)
                plt.close("all")
                gc.collect()
                window2.destroy()
                window.destroy()
                showinfo(title="Loaded file",message="Loaded pre-processed EEG file")
            Button(frame,text="OK",command=cont_load2).grid(row=len(eeg1.ch_names)+4,column=0,columnspan=2)    
        Button(window,text="OK",command=cont_load).grid(row=4,column=0,padx=10)

#load EEG montage (e.g. 10-20)
def load_montage():
    global raw1
    global raw2
    global events1
    global events2
    global event_dict
    global montage
    st_or_file=IntVar()
    st_or_file.set(1)
    sel_st_montage=StringVar()
    sel_st_montage.set('Standard 10-20')
    stlist=['Standard 10-20','Standard 10-10','Standard 10-05',
            'Biosemi 16','Biosemi 32','Biosemi 64','Biosemi 128',
            'Biosemi 160','Biosemi 160','Biosemi 256','Easycap M1',
            'Easycap M10','EGI 256','GSN HydroCel 32','GSN HydroCel 64',
            'GSN HydroCel 65','GSN HydroCel 128','GSN HydroCel 129',
            'GSN HydroCel 256','GSN HydroCel 257','MGH 60','MGH 70',
            'Artinis Octamon (fNIRS)','Artinis Brite23 (fNIRS)']
    st_montage_name=['cap1020','standard_1020','standard_1005',
                     'biosemi16','biosemi32','biosemi64','biosemi128',
                     'biosemi160','biosemi256','easycap-M1','easycap-M10',
                     'EGI_256','GSN-HydroCel-32','GSN-HydroCel-64_1.0',
                     'GSN-HydroCel-65_1.0','GSN-HydroCel-128','GSN-HydroCel-129',
                     'GSN-HydroCel-256','GSN-HydroCel-257','mgh60','mgh70',
                     'artinis-octamon','artinis-brite23']
    window=Toplevel(main)
    window.title('Select montage')
    window.geometry("")
    Label(window,text="Select montage:").grid(row=0,column=0,columnspan=2,padx=10,pady=10)
    Radiobutton(window,text="From standard ",variable=st_or_file,value=1).grid(row=1,column=0,padx=10,sticky=W)
    OptionMenu(window,sel_st_montage,*stlist).grid(row=1,column=1,sticky=W)
    Radiobutton(window,text="From file",variable=st_or_file,value=2).grid(row=2,column=0,padx=10,sticky=W)
    def applymontage():
        global raw1
        global raw2
        global events1
        global events2
        global event_dict
        global montage
        error=0
        if st_or_file.get()==1:
            for i in range(len(stlist)):
                if sel_st_montage.get()==stlist[i]:
                    chosen_montage=st_montage_name[i]
                    break
            if chosen_montage=='cap1020':
                montage=mne.channels.read_custom_montage('cap1020.txt')
            else:
                montage=mne.channels.make_standard_montage(chosen_montage)
        elif st_or_file.get()==2:
            try:
                filename=StringVar()
                filename.set(fd.askopenfilename(title="Load raw EEG",filetypes=(('All supported files',('*.loc','*.locs','*.eloc','*.sfp','*.csd','*.elc','*.txt','*.csd','*.elp','*.bvef','*.csv','*.tsv','*.xyz')),
                                                                                ('EEGLAB file',('*.loc','*.locs','*.eloc')),
                                                                                ('BESA/EGI file',('*.sfp')),
                                                                                ('BESA spherical file',('*.csd','*.elc','*.txt','*.csd','*.elp')),
                                                                                ('BrainVision file','*.bvef'),
                                                                                ('XYZ Coordinates',('*.csv','*.tsv','*.xyz')))))
                montage=mne.channels.read_custom_montage(filename.get())
            except:
                showinfo(title="Error",message="Could not load the file")
                error=1
        if error==0:
            if raw1 is not None:
                channel_list=['']+['Not connected']+raw1.ch_names
                event1name=StringVar()
                event2name=StringVar()
                event1name.set('quiet')
                event2name.set('motion')
                sel_correspondence=[]
                for i in range(len(montage.ch_names)):
                    sel_correspondence.append(StringVar())
                    con=0
                    for j in range(len(raw1.ch_names)):
                        if montage.ch_names[i] in raw1.ch_names[j]:
                            sel_correspondence[i].set(raw1.ch_names[j])
                            con=1
                            break
                    if con==0:
                        sel_correspondence[i].set('Not connected')
                stim_chan=StringVar()
                for j in range(len(raw1.ch_names)):
                    if ("DC03" in raw1.ch_names[j]) or ("DC3" in raw1.ch_names[j]):
                        stim_chan.set(raw1.ch_names[j])
                def onFrameConfigure(canvas):
                    canvas.configure(scrollregion=canvas.bbox("all"))
                window2=Toplevel(main)
                window2.geometry("675x600")
                window2.title("Match the electrodes")
                canvas=Canvas(window2,borderwidth=0,width=650,height=600)
                frame=Frame(canvas)
                scroll=Scrollbar(window2,orient="vertical",command=canvas.yview)
                canvas.configure(yscrollcommand=scroll.set)
                canvas.grid(row=0,column=0,columnspan=4)
                scroll.grid(row=0,column=4,sticky='ns')
                canvas.create_window((15,15), window=frame, anchor="nw")
                frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))
                Label(frame,text="Find the correspondence between the\nstandard electrode names and channel names on raw EEG file",justify=CENTER).grid(row=0,column=0,columnspan=4)
                Label(frame,text="Standard electrode names").grid(row=1,column=0,padx=10,sticky=W)
                Label(frame,text="Channel names on raw EEG file").grid(row=1,column=1,sticky=E)
                Label(frame,text="Event 1 name").grid(row=1,column=2,sticky=E,padx=10)
                Label(frame,text="Event 2 name").grid(row=1,column=3,sticky=E,padx=10)
                Label(frame,text="Event channel (stimulus sync)").grid(row=2,column=0,padx=10,sticky=W)
                OptionMenu(frame,stim_chan,*channel_list).grid(row=2,column=1,sticky=E)
                Entry(frame,textvariable=event1name,width=8).grid(row=2,column=2,sticky=E)
                Entry(frame,textvariable=event2name,width=8).grid(row=2,column=3,sticky=E)
                for i in range(len(montage.ch_names)):
                    Label(frame,text=montage.ch_names[i]).grid(row=i+3,column=0,padx=10,sticky=W)
                    OptionMenu(frame,sel_correspondence[i],*channel_list).grid(row=i+3,column=1,sticky=E)
                def update_chan():
                    global raw1
                    global raw2
                    global events1
                    global events2
                    global event_dict
                    def do_reordering():#this function is the channel reordering, which is called after sorting out the stimulus channel  
                        global raw1
                        global raw2
                        connected_chans=[i.get() for i in sel_correspondence if i.get()!='Not connected']
                        raw1=raw1.pick_channels(connected_chans)
                        connection_dic={sel_correspondence[i].get():montage.ch_names[i] for i in range(len(montage.ch_names)) if sel_correspondence[i].get()!='Not connected'}
                        raw1.rename_channels(mapping=connection_dic)
                        raw1.set_montage(montage)
                        if raw2 is not None:
                            raw2=raw2.pick_channels(connected_chans)
                            raw2.rename_channels(mapping=connection_dic)
                            raw2.set_montage(montage)
                        channel_order_list=list(range(1,len(raw1.ch_names)+1))
                        sel_chan_ord=[]
                        for i in range(len(raw1.ch_names)):
                            sel_chan_ord.append(StringVar())
                            sel_chan_ord[i].set(str(i+1))
                        window3=Toplevel(main)
                        window3.title("Reorder channels")
                        window3.geometry("225x400")
                        plt.close("all")
                        raw1.plot_sensors(show_names=True)
                        def onFrameConfigure(canvas):
                            canvas.configure(scrollregion=canvas.bbox("all"))
                        canvas=Canvas(window3,borderwidth=0,width=250,height=400)
                        frame=Frame(canvas)
                        scroll=Scrollbar(window3,orient="vertical",command=canvas.yview)
                        canvas.configure(yscrollcommand=scroll.set)
                        canvas.grid(row=0,column=0,columnspan=4)
                        scroll.grid(row=0,column=3,sticky='ns')
                        canvas.create_window((15,15), window=frame, anchor="nw")
                        frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))
                        Label(frame,text="Reorder channels (optional)",justify=CENTER).grid(row=0,column=0,columnspan=2)
                        Label(frame,text="Order").grid(row=1,column=0,padx=10,sticky=W)
                        Label(frame,text="Electrode channel").grid(row=1,column=1,sticky=E)
                        for i in range(len(raw1.ch_names)):
                            channel_order_list2=[channel_order_list[j] for j in range(i,len(channel_order_list))]+[channel_order_list[j] for j in range(i) if i!=0]
                            OptionMenu(frame,sel_chan_ord[i],*channel_order_list2).grid(row=i+2,column=0,sticky=W)
                            Label(frame,text=raw1.ch_names[i]).grid(row=i+2,column=1,padx=10,sticky=E)
                        def cont_load2():
                            global raw1
                            global raw2
                            chan_ord=[int(i.get())-1 for i in sel_chan_ord]
                            order=[raw1.ch_names[i] for i in chan_ord]
                            raw1=raw1.reorder_channels(order)
                            if raw2 is not None:
                                raw2=raw2.reorder_channels(order)
                            window3.destroy()
                            plt.close("all")
                            showinfo(title="Load complete", message="EEG montage loaded")
                        Button(frame,text="OK",command=cont_load2).grid(row=len(raw1.ch_names)+4,column=0,columnspan=2)
                        window2.destroy()
                    stim1=None
                    if stim_chan.get()!='Not connected' and stim_chan.get()!='': #this part within "if" is the automatic event detection from photodetector.
                        window2.destroy()
                        t1=raw1.times
                        stim1=raw1.get_data(picks=[stim_chan.get()])
                        plt.figure(1)
                        plt.scatter(t1,stim1[0],s=1,c='k')
                        plt.title("Condition "+cond1_name.get()+" stimulus channel")
                        plt.xlabel("Time (s)")
                        plt.ylabel("Channel values (a.u.)")
                        if raw2 is not None:
                            t2=raw2.times
                            stim2=raw2.get_data(picks=[stim_chan.get()])
                            plt.figure(2)
                            plt.scatter(t2,stim2[0],s=1,c='k')
                            plt.title("Condition "+cond2_name.get()+" stimulus channel")
                            plt.xlabel("Time (s)")
                            plt.ylabel("Channel values (a.u.)")
                        auto_threshold_ev1=np.mean([np.mean(stim1),np.quantile(stim1,0.9)])
                        auto_threshold_ev2=np.mean([np.mean(stim1),np.quantile(stim1,0.1)])
                        cond1_start=StringVar()
                        cond1_start.set(str(t1[0]))
                        cond1_end=StringVar()
                        cond1_end.set(str(t1[-1]))
                        cond1_ev1_opt=IntVar()
                        cond1_ev1_opt.set(2)
                        cond1_ev2_opt=IntVar()
                        cond1_ev2_opt.set(1)
                        cond1_ev1_leq=StringVar()
                        cond1_ev1_geq=StringVar()
                        cond1_ev1_geq.set(str(auto_threshold_ev1))
                        cond1_ev1_bt1=StringVar()
                        cond1_ev1_bt2=StringVar()
                        cond1_ev2_leq=StringVar()
                        cond1_ev2_leq.set(str(auto_threshold_ev2))
                        cond1_ev2_geq=StringVar()
                        cond1_ev2_bt1=StringVar()
                        cond1_ev2_bt2=StringVar()
                        cond1_ev_tmin=StringVar()
                        cond1_ev_tmin.set("1")
                        winstim=Toplevel(main)
                        winstim.title('Threshold for events')
                        Label(winstim,text="Select thresholds for events\nbased on stimulus channel").grid(row=0,column=0,columnspan=6)
                        Label(winstim,text="Condition "+cond1_name.get()).grid(row=1,column=0,padx=10,sticky=W)
                        Label(winstim,text="Experiment time range:").grid(row=2,column=0,padx=10,sticky=W)
                        Label(winstim,text="From: ").grid(row=2,column=1,sticky=W)
                        Entry(winstim,textvariable=cond1_start,width=8).grid(row=2,column=2,sticky=W)
                        Label(winstim,text="To: ").grid(row=2,column=3,sticky=W)
                        Entry(winstim,textvariable=cond1_end,width=8).grid(row=2,column=4,sticky=W)
                        Label(winstim,text="Event "+event1name.get()).grid(row=3,column=0,padx=10,sticky=W)
                        Label(winstim,text="Stimulus channel value ").grid(row=3,column=1,sticky=W)
                        Radiobutton(winstim,text="less than",variable=cond1_ev1_opt,value=1).grid(row=3,column=2,sticky=W)
                        Entry(winstim,textvariable=cond1_ev1_leq,width=8).grid(row=3,column=3,sticky=W)
                        Radiobutton(winstim,text="greater than",variable=cond1_ev1_opt,value=2).grid(row=4,column=2,sticky=W)
                        Entry(winstim,textvariable=cond1_ev1_geq,width=8).grid(row=4,column=3,sticky=W)
                        Radiobutton(winstim,text="between",variable=cond1_ev1_opt,value=3).grid(row=5,column=2,sticky=W)
                        Entry(winstim,textvariable=cond1_ev1_bt1,width=8).grid(row=5,column=3,sticky=W)
                        Label(winstim,text="and").grid(row=5,column=4)
                        Entry(winstim,textvariable=cond1_ev1_bt2,width=8).grid(row=5,column=5,sticky=W)
                        Label(winstim,text="Event "+event2name.get()).grid(row=7,column=0,padx=10,sticky=W)
                        Label(winstim,text="Stimulus channel value ").grid(row=7,column=1,sticky=W)
                        Radiobutton(winstim,text="less than",variable=cond1_ev2_opt,value=1).grid(row=7,column=2,sticky=W)
                        Entry(winstim,textvariable=cond1_ev2_leq,width=8).grid(row=7,column=3,sticky=W)
                        Radiobutton(winstim,text="greater than",variable=cond1_ev2_opt,value=2).grid(row=8,column=2,sticky=W)
                        Entry(winstim,textvariable=cond1_ev2_geq,width=8).grid(row=8,column=3,sticky=W)
                        Radiobutton(winstim,text="between",variable=cond1_ev2_opt,value=3).grid(row=9,column=2,sticky=W)
                        Entry(winstim,textvariable=cond1_ev2_bt1,width=8).grid(row=9,column=3,sticky=W)
                        Label(winstim,text="and").grid(row=9,column=4)
                        Entry(winstim,textvariable=cond1_ev2_bt2,width=8).grid(row=9,column=5,sticky=W)
                        Label(winstim,text="Minimum duration of events (seconds)").grid(row=10,column=0,padx=10,sticky=W,columnspan=2)
                        Entry(winstim,textvariable=cond1_ev_tmin,width=8).grid(row=10,column=2,sticky=W)
                        if raw2 is not None:
                            cond2_start=StringVar()
                            cond2_start.set(str(t2[0]))
                            cond2_end=StringVar()
                            cond2_end.set(str(t2[-1]))
                            cond2_ev1_opt=IntVar()
                            cond2_ev1_opt.set(2)
                            cond2_ev2_opt=IntVar()
                            cond2_ev2_opt.set(1)
                            cond2_ev1_leq=StringVar()
                            cond2_ev1_geq=StringVar()
                            cond2_ev1_geq.set(str(auto_threshold_ev1))
                            cond2_ev1_bt1=StringVar()
                            cond2_ev1_bt2=StringVar()
                            cond2_ev2_leq=StringVar()
                            cond2_ev2_leq.set(str(auto_threshold_ev2))
                            cond2_ev2_geq=StringVar()
                            cond2_ev2_bt1=StringVar()
                            cond2_ev2_bt2=StringVar()
                            cond2_ev_tmin=StringVar()
                            cond2_ev_tmin.set("1")
                            Label(winstim,text="Condition "+cond2_name.get()).grid(row=11,column=0,padx=10,sticky=W)
                            Label(winstim,text="Experiment time range:").grid(row=12,column=0,padx=10,sticky=W)
                            Label(winstim,text="From: ").grid(row=12,column=1,sticky=W)
                            Entry(winstim,textvariable=cond2_start,width=8).grid(row=12,column=2,sticky=W)
                            Label(winstim,text="To: ").grid(row=12,column=3,sticky=W)
                            Entry(winstim,textvariable=cond2_end,width=8).grid(row=12,column=4,sticky=W)
                            Label(winstim,text="Event "+event1name.get()).grid(row=13,column=0,padx=10,sticky=W)
                            Label(winstim,text="Stimulus channel value ").grid(row=13,column=1,sticky=W)
                            Radiobutton(winstim,text="less than",variable=cond2_ev1_opt,value=1).grid(row=13,column=2,sticky=W)
                            Entry(winstim,textvariable=cond2_ev1_leq,width=8).grid(row=13,column=3,sticky=W)
                            Radiobutton(winstim,text="greater than",variable=cond2_ev1_opt,value=2).grid(row=14,column=2,sticky=W)
                            Entry(winstim,textvariable=cond2_ev1_geq,width=8).grid(row=14,column=3,sticky=W)
                            Radiobutton(winstim,text="between",variable=cond2_ev1_opt,value=3).grid(row=15,column=2,sticky=W)
                            Entry(winstim,textvariable=cond2_ev1_bt1,width=8).grid(row=15,column=3,sticky=W)
                            Label(winstim,text="and").grid(row=15,column=4)
                            Entry(winstim,textvariable=cond2_ev1_bt2,width=8).grid(row=15,column=5,sticky=W)
                            Label(winstim,text="Event "+event2name.get()).grid(row=17,column=0,padx=10,sticky=W)
                            Label(winstim,text="Stimulus channel value ").grid(row=17,column=1,sticky=W)
                            Radiobutton(winstim,text="less than",variable=cond2_ev2_opt,value=1).grid(row=17,column=2,sticky=W)
                            Entry(winstim,textvariable=cond2_ev2_leq,width=8).grid(row=17,column=3,sticky=W)
                            Radiobutton(winstim,text="greater than",variable=cond2_ev2_opt,value=2).grid(row=18,column=2,sticky=W)
                            Entry(winstim,textvariable=cond2_ev2_geq,width=8).grid(row=18,column=3,sticky=W)
                            Radiobutton(winstim,text="between",variable=cond2_ev2_opt,value=3).grid(row=19,column=2,sticky=W)
                            Entry(winstim,textvariable=cond2_ev2_bt1,width=8).grid(row=19,column=3,sticky=W)
                            Label(winstim,text="and").grid(row=19,column=4)
                            Entry(winstim,textvariable=cond2_ev2_bt2,width=8).grid(row=19,column=5,sticky=W)
                            Label(winstim,text="Minimum duration of events (seconds)").grid(row=20,column=0,padx=10,columnspan=2,sticky=W)
                            Entry(winstim,textvariable=cond2_ev_tmin,width=8).grid(row=20,column=2,sticky=W)    
                        def compute_stim():
                            global raw1
                            global raw2
                            global events1
                            global events2
                            global events1_data
                            global events2_data
                            global event_dict
                            raw1=raw1.crop(tmin=float(cond1_start.get()),tmax=float(cond1_end.get()))
                            stim1vals=raw1.get_data(picks=[stim_chan.get()])
                            events1_data=stim1vals.copy()
                            stim1data=[[]]
                            for i in range(len(stim1vals[0])):
                                k=0
                                if cond1_ev1_opt.get()==1:
                                    if stim1vals[0][i]<=float(cond1_ev1_leq.get()):
                                        stim1data[0].append(1)
                                        k+=1
                                elif cond1_ev1_opt.get()==2:
                                    if stim1vals[0][i]>=float(cond1_ev1_geq.get()):
                                        stim1data[0].append(1)
                                        k+=1
                                elif cond1_ev1_opt.get()==3:
                                    if float(cond1_ev1_bt1.get())<=stim1vals[0][i]<=float(cond1_ev1_bt2.get()):
                                        stim1data[0].append(1)
                                        k+=1
                                if cond1_ev2_opt.get()==1:
                                    if stim1vals[0][i]<=float(cond1_ev2_leq.get()):
                                        stim1data[0].append(2)
                                        k+=1
                                elif cond1_ev2_opt.get()==2:
                                    if stim1vals[0][i]>=float(cond1_ev2_geq.get()):
                                        stim1data[0].append(1)
                                        k+=1
                                elif cond1_ev2_opt.get()==3:
                                    if float(cond1_ev2_bt1.get())<=stim1vals[0][i]<=float(cond1_ev2_bt2.get()):
                                        stim1data[0].append(1)
                                        k+=1
                                if k==2:
                                    stim1data[0].pop(-1)
                                elif k==0:
                                    stim1data[0].append(0)
                            info = mne.create_info(['STI'], raw1.info['sfreq'], ['stim'])
                            stim1data = mne.io.RawArray(stim1data,info)
                            events1=mne.find_events(stim1data,consecutive=False,min_duration=float(cond1_ev_tmin.get()))
                            event_dict={event1name.get():1,event2name.get():2}
                            if raw2 is not None:
                                raw2=raw2.crop(tmin=float(cond2_start.get()),tmax=float(cond2_end.get()))
                                stim2vals=raw2.get_data(picks=[stim_chan.get()])
                                events2_data=stim2vals.copy()
                                stim2data=[[]]
                                for i in range(len(stim2vals[0])):
                                    k=0
                                    if cond2_ev1_opt.get()==1:
                                        if stim2vals[0][i]<=float(cond2_ev1_leq.get()):
                                            stim2data[0].append(1)
                                            k+=1
                                    elif cond2_ev1_opt.get()==2:
                                        if stim2vals[0][i]>=float(cond2_ev1_geq.get()):
                                            stim2data[0].append(1)
                                            k+=1
                                    elif cond2_ev1_opt.get()==3:
                                        if float(cond2_ev1_bt1.get())<=stim2vals[0][i]<=float(cond2_ev1_bt2.get()):
                                            stim2data[0].append(1)
                                            k+=1
                                    if cond2_ev2_opt.get()==1:
                                        if stim2vals[0][i]<=float(cond2_ev2_leq.get()):
                                            stim2data[0].append(2)
                                            k+=1
                                    elif cond2_ev2_opt.get()==2:
                                        if stim2vals[0][i]>=float(cond2_ev2_geq.get()):
                                            stim2data[0].append(1)
                                            k+=1
                                    elif cond2_ev2_opt.get()==3:
                                        if float(cond2_ev2_bt1.get())<=stim2vals[0][i]<=float(cond2_ev2_bt2.get()):
                                            stim2data[0].append(1)
                                            k+=1
                                    if k==2:
                                        stim2data[0].pop(-1)
                                    elif k==0:
                                        stim2data[0].append(0)
                                info2 = mne.create_info(['STI'], raw2.info['sfreq'], ['stim'])
                                stim2data = mne.io.RawArray(stim2data,info2)
                                events2=mne.find_events(stim2data,consecutive=False,min_duration=float(cond2_ev_tmin.get()))
                            winstim.destroy()
                            do_reordering()    
                        Button(winstim,text="OK",command=compute_stim).grid(row=21,column=0,columnspan=6)
                        plt.show()
                    else:
                        do_reordering()
                Button(frame,text="OK",command=update_chan).grid(row=len(montage.ch_names)+4,column=0,columnspan=4)
            if raw1 is None:
                showinfo(title="Error",message="Please load raw EEG file(s)\nbefore selecting the montage.")
        window.destroy()
    Button(window,text="OK",command=applymontage).grid(row=3,column=0,columnspan=2)

#preprocessing pipeline
def preprocess():
    global raw1
    global raw2
    if raw1 is None:
        showinfo(title="Error",message="At least 1 raw (edf) data is required,\nalong with EEG montage.") 
    else:
        #delete bad channels
        showinfo(title="Info",message="Observe the PSDs and inform\nthe channels to be excluded")
        chan2del=StringVar()
        win=Toplevel()
        win.title("Channels to delete")
        Label(win,text="Select channels to delete (separate with commas):").grid(row=0,column=0,padx=10)
        Entry(win,textvariable=chan2del,width=50).grid(row=1,column=0,padx=10)
        def step1():
            global raw1
            global raw2
            win.destroy()
            plt.close("all")
            bads=chan2del.get().replace(" ","").split(',')
            if bads[0]!='':
                raw1.drop_channels(bads)
                if raw2 is not None:
                    raw2.drop_channels(bads)
            fmin=StringVar()
            fmin.set("1")
            fmax=StringVar()
            fmax.set("50")
            #delete DBS spikes
            hasdbs=IntVar()
            hasdbs.set(2)
            win2=Toplevel()
            win2.lift()
            win2.title("Filtering")
            Label(win2,text="Select filter parameters:").grid(row=0,column=0,padx=10,columnspan=3)
            Label(win2,text="Minimum frequency (Hz):").grid(row=1,column=0,padx=10,sticky=W)
            Entry(win2,textvariable=fmin,width=8).grid(row=1,column=1,sticky=W)
            Label(win2,text="Maximum frequency (Hz):").grid(row=2,column=0,padx=10,sticky=W)
            Entry(win2,textvariable=fmax,width=8).grid(row=2,column=1,sticky=W)
            Label(win2,text="Does any of the conditions contain DBS\nor frequency spike artifacts?:").grid(row=3,column=0,padx=10,sticky=W,columnspan=3)
            Radiobutton(win2,text="Yes, condition "+cond1_name.get(),variable=hasdbs,value=1).grid(row=4,column=1,padx=10,sticky=W,columnspan=3)
            Radiobutton(win2,text="Yes, condition "+cond2_name.get(),variable=hasdbs,value=2).grid(row=5,column=1,padx=10,sticky=W,columnspan=3)
            Radiobutton(win2,text="Yes, both conditions",variable=hasdbs,value=3).grid(row=6,column=1,padx=10,sticky=W,columnspan=3)
            Radiobutton(win2,text="No",variable=hasdbs,value=4).grid(row=7,column=1,padx=10,sticky=W,columnspan=3)
            def step2():
                global raw1
                global raw2
                raw1=raw1.filter(float(fmin.get()),float(fmax.get()))
                if raw2 is not None:
                    raw2=raw2.filter(float(fmin.get()),float(fmax.get()))
                def step3():
                    do_ica=IntVar()
                    do_ica.set(1)
                    win4=Toplevel()
                    win4.lift()
                    win4.title("Perform ICA")
                    Label(win4,text="Would you like to perform Independent Component Analysis\nto identify and remove eye blinks, saccades, motor and noise artifacts?\n(not recommended in case of low memory)").grid(row=0,column=0,columnspan=2,padx=10)
                    Radiobutton(win4,text="Yes",variable=do_ica,value=1).grid(row=1,column=0,padx=10)
                    Radiobutton(win4,text="No",variable=do_ica,value=2).grid(row=1,column=1,padx=10)
                    def step4():
                        win4.destroy()
                        def step5():
                            #Re-referencing
                            ref_opt=IntVar()
                            ref_opt.set(1)
                            chan_ref_exc=StringVar()
                            chan_ref_inc=StringVar()
                            win8=Toplevel()
                            win8.title("Re-referencing")
                            Label(win8,text="Would you like to re-reference the channels according to:").grid(row=0,column=0,padx=10,columnspan=2)
                            Radiobutton(win8,text="Average of all channels",variable=ref_opt,value=1).grid(row=1,column=0,padx=10,sticky=W)
                            Radiobutton(win8,text="Average of all channels, except:",variable=ref_opt,value=2).grid(row=2,column=0,padx=10,sticky=W)
                            Entry(win8,textvariable=chan_ref_exc,width=20).grid(row=2,column=1,padx=10,sticky=W)
                            Radiobutton(win8,text="Average of the following channels:",variable=ref_opt,value=3).grid(row=3,column=0,padx=10,sticky=W)
                            Entry(win8,textvariable=chan_ref_inc,width=20).grid(row=3,column=1,padx=10,sticky=W)
                            Radiobutton(win8,text="Using Reference Electrode Standardization Technique (REST) to a point at infinity",variable=ref_opt,value=4).grid(row=4,column=0,padx=10,sticky=W,columnspan=2)
                            Radiobutton(win8,text="Do not perform re-referencing",variable=ref_opt,value=5).grid(row=5,column=0,padx=10,sticky=W)
                            def do_reref():
                                global raw1
                                global raw2
                                plt.close("all")
                                if ref_opt.get()==1:
                                    raw1=raw1.copy().set_eeg_reference()
                                    if raw2 is not None:
                                        raw2=raw2.copy().set_eeg_reference()
                                elif ref_opt.get()==2:
                                    excl_chan=chan_ref_exc.get().replace(" ","").split(',')
                                    ref_chans=[chan for chan in raw1.ch_names if chan not in excl_chan]
                                    raw1=raw1.copy().set_eeg_reference(ref_chans)
                                    if raw2 is not None:
                                        raw2=raw2.copy().set_eeg_reference(ref_chans)
                                elif ref_opt.get()==3:
                                    ref_chans=chan_ref_inc.get().replace(" ","").split(',')
                                    raw1=raw1.copy().set_eeg_reference(ref_chans)
                                    if raw2 is not None:
                                        raw2=raw2.copy().set_eeg_reference(ref_chans)
                                elif ref_opt.get()==4:
                                    raw1.del_proj()
                                    sphere1 = mne.make_sphere_model('auto', 'auto', raw1.info)
                                    src1 = mne.setup_volume_source_space(sphere=sphere, exclude=30., pos=15.)
                                    forward1 = mne.make_forward_solution(raw1.info, trans=None, src=src, bem=sphere)
                                    raw1=raw1.copy().set_eeg_reference('REST',forward=forward1)
                                    if raw2 is not None:
                                        raw2.del_proj()
                                        sphere2 = mne.make_sphere_model('auto', 'auto', raw2.info)
                                        src2 = mne.setup_volume_source_space(sphere=sphere, exclude=30., pos=15.)
                                        forward1 = mne.make_forward_solution(raw1.info, trans=None, src=src, bem=sphere)
                                        raw2=raw2.copy().set_eeg_reference('REST',forward=forward2)
                                win8.destroy()
                                showinfo(title="Info",message="We will now save this\nprocessed continuous EEG data.\nEvents will be stored as an additional EEG channel.\nAfter this, we will do epoching\n(splitting the EEG in trials)")
                                if events1_data is not None:
                                    fname1 = fd.asksaveasfilename(title="EEG condition "+cond1_name.get(),defaultextension=".edf",filetypes=(("European Data Format (EDF)", "*.edf"),("All Files", "*.*")))
                                    info1 = mne.create_info(raw1.ch_names+['Events'], ch_types=['eeg']*(len(raw1.ch_names)+1), sfreq=raw1.info['sfreq'])
                                    data1=raw1.get_data()
                                    data1=np.vstack((data1,events1_data))
                                    raw1_to_save=mne.io.RawArray(data1, info1)
                                    mne.export.export_raw(fname1, raw1_to_save, fmt='auto',overwrite='True')
                                    if raw2 is not None:
                                        fname2 = fd.asksaveasfilename(title="EEG condition "+cond2_name.get(),defaultextension=".edf",filetypes=(("European Data Format (EDF)", "*.edf"),("All Files", "*.*")))
                                        info2 = mne.create_info(raw2.ch_names+['Events'], ch_types=['eeg']*(len(raw2.ch_names)+1), sfreq=raw2.info['sfreq'])
                                        data2=raw2.get_data()
                                        data2=np.vstack((data2,events2_data))                                       
                                        raw2_to_save=mne.io.RawArray(data2, info2)
                                        mne.export.export_raw(fname2, raw2_to_save, fmt='auto',overwrite='True')
                                else:
                                    fname1 = fd.asksaveasfilename(title="EEG condition "+cond1_name.get(),defaultextension=".edf",filetypes=(("European Data Format (EDF)", "*.edf"),("All Files", "*.*")))
                                    mne.export.export_raw(fname1, raw1, fmt='auto',overwrite='True')
                                    if raw2 is not None:
                                        fname2 = fd.asksaveasfilename(title="EEG condition "+cond2_name.get(),defaultextension=".edf",filetypes=(("European Data Format (EDF)", "*.edf"),("All Files", "*.*")))
                                        mne.export.export_raw(fname2, raw2, fmt='auto',overwrite='True')
                                def step6():
                                    #Epoching
                                    whole_part_event=IntVar()
                                    whole_part_event.set("1")
                                    duration_whole=StringVar()
                                    duration_whole.set(str(int(np.median(np.diff(events1[:,0]))/raw1.info['sfreq'])))
                                    tstart_whole=StringVar()
                                    tstart_whole.set("0")
                                    duration_part=StringVar()
                                    duration_part.set("5")
                                    eventduration_part=StringVar()
                                    eventduration_part.set(str(int(np.median(np.diff(events1[:,0]))/raw1.info['sfreq'])))
                                    tstart_part=StringVar()
                                    tstart_part.set("0")
                                    reject_criteria=StringVar()
                                    if raw2 is not None:
                                        reject_criteria.set(f'{max([np.quantile(np.abs(raw1.get_data()),0.999),np.quantile(np.abs(raw2.get_data()),0.999)])*10**6:.2f}')
                                    else:
                                        reject_criteria.set(f'{np.quantile(np.abs(raw1.get_data()),0.999)*10**6:.2f}')
                                    flat_criteria=StringVar()
                                    flat_criteria.set("1")
                                    win9=Toplevel()
                                    win9.title("Epoching")
                                    Label(win9,text="Epoching (division of EEG in trials)\nSelect parameters").grid(row=0,column=0,columnspan=5,padx=10)
                                    Separator(win9,orient='horizontal').grid(row=1,column=0,columnspan=5,sticky='new',padx=10)
                                    Radiobutton(win9,text="Whole event",variable=whole_part_event,value=1).grid(row=2,column=0,sticky=W,padx=10)
                                    Label(win9,text="Epoch start time relative to event (s):").grid(row=3,column=0,columnspan=2,sticky=W,padx=10)
                                    Entry(win9,textvariable=tstart_whole).grid(row=3,column=1,sticky=W,padx=10)
                                    Label(win9,text="Epoch duration (s):").grid(row=4,column=0,sticky=W,padx=10)
                                    Entry(win9,textvariable=duration_whole).grid(row=4,column=1,sticky=W,padx=10)
                                    Separator(win9,orient='horizontal').grid(row=5,column=0,columnspan=5,sticky='new',padx=10)
                                    Radiobutton(win9,text="Subset of event (for long stimulus only)",variable=whole_part_event,value=2).grid(row=6,column=0,sticky=W,padx=10)
                                    Label(win9,text="Epoch start time relative to event (s):").grid(row=7,column=0,columnspan=2,sticky=W,padx=10)
                                    Entry(win9,textvariable=tstart_part).grid(row=7,column=1,sticky=W,padx=10)
                                    Label(win9,text="Epoch duration (s):").grid(row=8,column=0,sticky=W,padx=10)
                                    Entry(win9,textvariable=duration_part).grid(row=8,column=1,sticky=W,padx=10)
                                    Label(win9,text="Event duration (s):").grid(row=9,column=0,sticky=W,padx=10)
                                    Entry(win9,textvariable=eventduration_part).grid(row=9,column=1,sticky=W,padx=10)
                                    Separator(win9,orient='horizontal').grid(row=10,column=0,columnspan=5,sticky='new',padx=10)
                                    Label(win9,text="Automatic epoch rejection criteria:").grid(row=11,column=0,columnspan=2,sticky=W,padx=10)
                                    Label(win9,text="Reject epoch with signal above (\u03bcV):").grid(row=12,column=0,columnspan=2,sticky=W,padx=10)
                                    Entry(win9,textvariable=reject_criteria).grid(row=12,column=1,sticky=W,padx=10)
                                    Label(win9,text="Reject epoch of signal below (\u03bcV) (flat criteria):").grid(row=13,column=0,columnspan=2,sticky=W,padx=10)
                                    Entry(win9,textvariable=flat_criteria).grid(row=13,column=1,sticky=W,padx=10)
                                    def epoching():
                                        global epoched_raw1
                                        global epoched_raw2
                                        def step7():
                                            #Manually inspect epochs
                                            global epoched_raw1
                                            global epoched_raw2
                                            manual_inspect=IntVar()
                                            manual_inspect.set(2)
                                            win10=Toplevel()
                                            win10.title("Epochs: manual inspection")
                                            Label(win10,text="Would you like to manually inspect\neach epoch to find and reject\nthose with remaining artifacts?").grid(row=0,column=0,columnspan=2,padx=10)
                                            Radiobutton(win10,text="Yes",variable=manual_inspect,value=1).grid(row=1,column=0,padx=10)
                                            Radiobutton(win10,text="No",variable=manual_inspect,value=2).grid(row=1,column=1,padx=10)
                                            def inspect_epoch():
                                                global eeg1
                                                global eeg2
                                                global epoched_raw1
                                                global epoched_raw2
                                                global epochs_delete
                                                if manual_inspect.get()==1:
                                                    os.mkdir("epochs_inspect_cond1")
                                                    for epoch in range(len(epoched_raw1)):
                                                        fig1=epoched_raw1[epoch].plot(show=False)
                                                        fig1.savefig(f"epochs_inspect_cond1/{epoch:04d}.jpg")
                                                        plt.close("all")
                                                        gc.collect()
                                                    epochs_delete=StringVar()
                                                    win10.destroy()
                                                    win11=Toplevel()
                                                    win11.title("Epochs to delete - "+cond1_name.get())
                                                    Label(win11,text="Epochs, condition "+cond1_name.get()).grid(row=0,column=0,padx=10)
                                                    Label(win11,text="Inspect epochs on folder epochs_inspect_cond1.\nInform epochs to delete (separate with commas):").grid(row=1,column=0,padx=10)
                                                    Entry(win11,textvariable=epochs_delete,width=50).grid(row=2,column=0,padx=10)
                                                    def select_epochs1():
                                                        global eeg1
                                                        global eeg2
                                                        global epoched_raw1
                                                        global epoched_raw2
                                                        global epochs_delete
                                                        to_remove=epochs_delete.get().replace(" ","").split(',')
                                                        to_remove=[int(i) for i in to_remove]
                                                        to_stay=list(range(len(epoched_raw1)))
                                                        to_stay=[i for i in to_stay if i not in to_remove]
                                                        epoched_raw1=epoched_raw1[to_stay]
                                                        eeg1=epoched_raw1
                                                        win11.destroy()
                                                        if raw2 is not None:
                                                            os.mkdir("epochs_inspect_cond2")
                                                            shutil.rmtree("epochs_inspect_cond1")
                                                            for epoch in range(len(epoched_raw2)):
                                                                fig1=epoched_raw2[epoch].plot(show=False)
                                                                fig1.savefig(f"epochs_inspect_cond2/{epoch:04d}.jpg")
                                                                plt.close("all")
                                                                gc.collect()
                                                            epochs_delete=StringVar()
                                                            win12=Toplevel()
                                                            win12.title("Epochs to delete - "+cond1_name.get())
                                                            Label(win12,text="Epochs, condition "+cond1_name.get()).grid(row=0,column=0,padx=10)
                                                            Label(win12,text="Inspect epochs on folder epochs_inspect_cond2.\nInform epochs to delete (separate with commas):").grid(row=1,column=0,padx=10)
                                                            Entry(win12,textvariable=epochs_delete,width=50).grid(row=2,column=0,padx=10)
                                                            def select_epochs2():
                                                                global eeg2
                                                                global epoched_raw2
                                                                global epochs_delete
                                                                to_remove=epochs_delete.get().replace(" ","").split(',')
                                                                to_remove=[int(i) for i in to_remove]
                                                                to_stay=list(range(len(epoched_raw2)))
                                                                to_stay=[i for i in to_stay if i not in to_remove]
                                                                epoched_raw2=epoched_raw2[to_stay]
                                                                eeg2=epoched_raw2
                                                                shutil.rmtree("epochs_inspect_cond2")
                                                                win12.destroy()
                                                                save_preprocessed()
                                                                showinfo(title="Info",message="Pre-processing completed")
                                                            Button(win12,text="OK",command=select_epochs2).grid(row=3,column=0,padx=10)
                                                        else:
                                                            save_preprocessed()
                                                            showinfo(title="Info",message="Pre-processing completed")
                                                    Button(win11,text="OK",command=select_epochs1).grid(row=3,column=0,padx=10)
                                                else:
                                                    eeg1=epoched_raw1
                                                    if raw2 is not None:
                                                        eeg2=epoched_raw2
                                                    win10.destroy()
                                                    save_preprocessed()
                                                    showinfo(title="Info",message="Pre-processing completed")
                                            def save_preprocessed():
                                                showinfo(title="Info",message="Next, we will save the pre-processed EEG data")
                                                fname1 = fd.asksaveasfilename(title="Pre-processed EEG "+cond1_name.get()+"_epo.fif",defaultextension="_epo.fif",filetypes=(("Epoched FIF file", "*_epo.fif"),("All Files", "*.*")))
                                                eeg1.save(fname1,overwrite=True)
                                                if raw2 is not None:
                                                    fname2 = fd.asksaveasfilename(title="Pre-processed EEG "+cond2_name.get()+"_epo.fif",defaultextension="_epo.fif",filetypes=(("Epoched FIF file", "*_epo.fif"),("All Files", "*.*")))
                                                    eeg2.save(fname2,overwrite=True)
                                            Button(win10,text="OK",command=inspect_epoch).grid(row=2,column=0,columnspan=2,padx=10)
                                        if whole_part_event.get()==2:
                                            split_times=[(i,i+float(duration_part.get())) for i in np.arange(float(tstart_part.get()),float(eventduration_part.get()),float(duration_part.get())) if i+float(duration_part.get())<=float(eventduration_part.get())]
                                            split_raw1=[]
                                            for (tmin,tmax) in split_times:
                                                split_raw1.append(mne.Epochs(raw1,events1,tmin=tmin,tmax=tmax,event_id=event_dict,preload=True,baseline=None))
                                                split_raw1[-1].drop_bad(reject=dict(eeg=float(reject_criteria.get())*10**-6),flat=dict(eeg=float(flat_criteria.get())*10**-6))
                                                if len(split_raw1)>1:
                                                    split_raw1[-1].shift_time(-(tmin-float(tstart_part.get())))
                                            epoched_raw1=mne.concatenate_epochs(split_raw1)
                                            epoched_raw1.load_data()
                                            if raw2 is not None:
                                                split_raw2=[]
                                                for (tmin,tmax) in split_times:
                                                    split_raw2.append(mne.Epochs(raw2,events2,tmin=tmin,tmax=tmax,event_id=event_dict,preload=True,baseline=None))
                                                    split_raw2[-1].drop_bad(reject=dict(eeg=float(reject_criteria.get())*10**-6),flat=dict(eeg=float(flat_criteria.get())*10**-6))
                                                    if len(split_raw2)>1:
                                                        split_raw2[-1].shift_time(-(tmin-float(tstart_part.get())))
                                                epoched_raw2=mne.concatenate_epochs(split_raw2)
                                                epoched_raw2.load_data()
                                            win9.destroy()
                                            step7()
                                        else:
                                            epoched_raw1=mne.Epochs(raw1,events1,tmin=float(tstart_whole.get()),tmax=float(duration_whole.get())+float(tstart_whole.get()),event_id=event_dict,preload=True,baseline=None)
                                            if raw2 is not None:
                                                epoched_raw2=mne.Epochs(raw2,events2,tmin=float(tstart_whole.get()),tmax=float(duration_whole.get())+float(tstart_whole.get()),event_id=event_dict,preload=True,baseline=None)
                                            win9.destroy()
                                            step7()
                                    Button(win9,text="OK",command=epoching).grid(row=14,column=0,padx=10,columnspan=5)
                                step6()
                            Button(win8,text="OK",command=do_reref).grid(row=6,column=0,padx=10,columnspan=2)
                            raw1.plot_sensors(show_names=True)
                        if do_ica.get()==1:
                            #ICA analysis and filtering
                            showinfo(title="Info",message="It will take a while to calculate ICA,\nplease be patient")
                            ica = mne.preprocessing.ICA(n_components=len(raw1.ch_names),random_state=50)
                            ica.fit(raw1)
                            win5=Toplevel()
                            win5.lift()
                            win5.title("ICA components to delete")
                            ica_to_delete=StringVar()
                            Label(win5,text="ICA, condition "+cond1_name.get()+", 1st run").grid(row=0,column=0,padx=10)
                            Label(win5,text="Check ICA components on folder ICA-cond1-round1.\nInform ICA components to delete (separate with commas):").grid(row=1,column=0,padx=10)
                            Entry(win5,textvariable=ica_to_delete,width=50).grid(row=2,column=0,padx=10)
                            def cont_ica():
                                temp=raw1.copy()
                                to_remove=ica_to_delete.get().replace(" ","").split(',')
                                if to_remove[0]!='':
                                  to_remove=[int(to_remove[i]) for i in range(len(to_remove))]
                                  ica.exclude=to_remove
                                  ica.apply(temp)
                                win5.destroy()
                                plt.close("all")
                                gc.collect()
                                showinfo(title="Info",message="It will take a while to calculate ICA,\nplease be patient")
                                ica2 = mne.preprocessing.ICA(n_components=len(temp.ch_names),random_state=50)
                                ica2.fit(temp)
                                win6=Toplevel()
                                win6.lift()
                                win6.title("ICA components to delete")
                                ica2_to_delete=StringVar()
                                Label(win6,text="ICA, condition "+cond1_name.get()+", 2nd run").grid(row=0,column=0,padx=10)
                                Label(win6,text="Check ICA components on folder ICA-cond1-round2.\nInform ICA components to delete (separate with commas):").grid(row=1,column=0,padx=10)
                                Entry(win6,textvariable=ica2_to_delete,width=50).grid(row=2,column=0,padx=10)
                                def end_ica():
                                  global raw1
                                  to_remove2=ica2_to_delete.get().replace(" ","").split(',')
                                  if to_remove2[0]!='':
                                    to_remove2=[int(to_remove2[i]) for i in range(len(to_remove2))]
                                    ica2.exclude=to_remove2
                                    ica2.apply(raw1)
                                  win6.destroy()
                                  plt.close("all")
                                  gc.collect()
                                  if raw2 is not None:
                                    showinfo(title="Info",message="It will take a while to calculate ICA,\nplease be patient")
                                    ica_2 = mne.preprocessing.ICA(n_components=len(raw2.ch_names),random_state=50)
                                    ica_2.fit(raw1)
                                    win7=Toplevel()
                                    win7.lift()
                                    win7.title("ICA components to delete")
                                    ica_to_delete2=StringVar()
                                    Label(win7,text="ICA, condition "+cond2_name.get()+", 1st run").grid(row=0,column=0,padx=10)
                                    Label(win7,text="Check ICA components on folder ICA-cond2-round1.\nInform ICA components to delete (separate with commas):").grid(row=1,column=0,padx=10)
                                    Entry(win7,textvariable=ica_to_delete2,width=50).grid(row=2,column=0,padx=10)
                                    def cont_ica2():
                                        temp2=raw2.copy()
                                        to_remove2_2=ica_to_delete2.get().replace(" ","").split(',')
                                        if to_remove2_2[0]!='':
                                          to_remove2_2=[int(to_remove2_2[i]) for i in range(len(to_remove2_2))]
                                          ica_2.exclude=to_remove2_2
                                          ica_2.apply(temp2)
                                        win7.destroy()
                                        plt.close("all")
                                        gc.collect()
                                        showinfo(title="Info",message="It will take a while to calculate ICA,\nplease be patient")
                                        ica2_2 = mne.preprocessing.ICA(n_components=len(temp2.ch_names),random_state=50)
                                        ica2_2.fit(temp2)
                                        win8=Toplevel()
                                        win8.lift()
                                        win8.title("ICA components to delete")
                                        ica2_to_delete2=StringVar()
                                        Label(win8,text="ICA, condition "+cond2_name.get()+", 2nd run").grid(row=0,column=0,padx=10)
                                        Label(win8,text="Check ICA components on folder ICA-cond2-round2.\nInform ICA components to delete (separate with commas):").grid(row=1,column=0,padx=10)
                                        Entry(win8,textvariable=ica2_to_delete2,width=50).grid(row=2,column=0,padx=10)
                                        def end_ica2():
                                            global raw2
                                            to_remove2_3=ica2_to_delete2.get().replace(" ","").split(',')
                                            if to_remove2_3[0]!='':
                                              to_remove2_3=[int(to_remove2_3[i]) for i in range(len(to_remove2))]
                                              ica2_2.exclude=to_remove2_3
                                              ica2_2.apply(raw2)
                                            win8.destroy()
                                            plt.close("all")
                                            gc.collect()
                                            shutil.rmtree("ICA-cond2-round2")
                                            showinfo(title="info",message="ICA filtering completed\nProceeding with re-referencing")
                                            step5()
                                        Button(win8,text="OK",command=end_ica2).grid(row=3,column=0,padx=10)
                                        os.mkdir("ICA-cond2-round2")
                                        shutil.rmtree("ICA-cond2-round1")
                                        ica22src=ica2_2.plot_sources(temp2,show=False)
                                        ica22prp=ica2_2.plot_properties(temp2,picks=list(range(len(temp2.ch_names))),show=False)
                                        ica22src.savefig("ICA-cond2-round2/icasrc.jpg")
                                        for i in range(len(ica22prp)):
                                            ica22prp[i].savefig("ICA-cond2-round2/ica"+str(i)+".jpg")
                                    Button(win7,text="OK",command=cont_ica2).grid(row=3,column=0,padx=10)
                                    os.mkdir("ICA-cond2-round1")
                                    shutil.rmtree("ICA-cond1-round2")
                                    ica2src=ica_2.plot_sources(raw2,show=False)
                                    ica2prp=ica_2.plot_properties(raw2,picks=list(range(len(raw2.ch_names))),show=False)
                                    ica2src.savefig("ICA-cond2-round1/icasrc.jpg")
                                    for i in range(len(ica2prp)):
                                        ica2prp[i].savefig("ICA-cond2-round1/ica"+str(i)+".jpg")
                                  else:
                                    showinfo(title="info",message="ICA filtering completed\nProceeding with re-referencing")
                                    step5()  
                                Button(win6,text="OK",command=end_ica).grid(row=3,column=0,padx=10)
                                os.mkdir("ICA-cond1-round2")
                                shutil.rmtree("ICA-cond1-round1")
                                ica12src=ica2.plot_sources(temp,show=False)
                                ica12prp=ica2.plot_properties(temp,picks=list(range(len(temp.ch_names))),show=False)
                                ica12src.savefig("ICA-cond1-round2/icasrc.jpg")
                                for i in range(len(ica12prp)):
                                    ica12prp[i].savefig("ICA-cond1-round2/ica"+str(i)+".jpg")
                            Button(win5,text="OK",command=cont_ica).grid(row=3,column=0,padx=10)
                            try:
                                os.mkdir("ICA-cond1-round1")
                            except:
                                shutil.rmtree("ICA-cond1-round1")
                                os.mkdir("ICA-cond1-round1")
                            icasrc=ica.plot_sources(raw1,show=False)
                            icaprp=ica.plot_properties(raw1,picks=list(range(len(raw1.ch_names))),show=False)
                            icasrc.savefig("ICA-cond1-round1/icasrc.jpg")
                            for i in range(len(icaprp)):
                                icaprp[i].savefig("ICA-cond1-round1/ica"+str(i)+".jpg")
                        else:
                            step5()
                    Button(win4,text="OK",command=step4).grid(row=2,column=0,padx=10,columnspan=2)
                #Delete DBS peaks:
                if hasdbs.get()==1 or hasdbs.get()==3:
                    fpeakstart1=StringVar()
                    fpeakstart1.set("8")
                    quantile1=StringVar()
                    quantile1.set("10")
                    win2.destroy()
                    win3=Toplevel()
                    win3.lift()
                    win3.title("Find peaks")
                    Label(win3,text="Select parameters to find peaks for condition "+cond1_name.get()).grid(row=0,column=0,padx=10,columnspan=3)
                    Label(win3,text="Frequency to start finding peaks (Hz):").grid(row=1,column=0,padx=10,sticky=W)
                    Entry(win3,textvariable=fpeakstart1,width=8).grid(row=1,column=1,sticky=W)
                    Label(win3,text="Sensitivity (0.0 to 10000):").grid(row=2,column=0,padx=10,sticky=W)
                    Entry(win3,textvariable=quantile1,width=8).grid(row=2,column=1,sticky=W)
                    def peakfind1():
                        plt.close("all")
                        psds, freqs = mne.time_frequency.psd_multitaper(raw1)
                        psds_mean=psds.mean(0)
                        f2=freqs[freqs>float(fpeakstart1.get())]
                        sig=psds_mean[np.where(freqs>float(fpeakstart1.get()))]
                        pk,prop=find_peaks(sig,distance=100,prominence=np.quantile(sig,1-float(quantile1.get())/10000))
                        wd,wdht,left_ips,right_ips=peak_widths(sig,pk)
                        sig2=10*np.log10(sig)
                        wdht2=10*np.log10(wdht)
                        plt.plot(f2,sig2)
                        plt.plot(f2[pk],sig2[pk],'x')
                        f_left=[f2[int(i)] for i in left_ips]
                        f_right=[f2[int(i)] for i in right_ips]
                        plt.hlines(wdht2,f_left,f_right,color="C2")
                        plt.xlabel("Frequency(Hz)")
                        plt.ylabel("PSD (dB)")
                        wd2=np.array([(f_right[i]-f_left[i])*2 for i in range(len(left_ips))])
                        peakbtn["text"]="Redo find peaks"
                        def deletepeaks1():
                            global raw1
                            filt_raw=mne.filter.notch_filter(raw1.get_data(),raw1.info['sfreq'],f2[pk],fir_window='hann',notch_widths=wd2)
                            raw1=mne.io.RawArray(filt_raw,raw1.info)
                            win3.destroy()
                            plt.close("all")
                            showinfo(title="Info",message="Selected frequency peaks\ndeleted in condition\n"+cond1_name.get())
                            if hasdbs.get()==1:
                                step3()
                        Button(win3,text="Peaks correctly found\nproceed with filter",command=deletepeaks1).grid(row=11,column=0,columnspan=3)
                        plt.show()
                    peakbtn=Button(win3,text="Find peaks",command=peakfind1)
                    peakbtn.grid(row=10,column=0,columnspan=3)
                if hasdbs.get()==2 or hasdbs.get()==3:
                    fpeakstart2=StringVar()
                    fpeakstart2.set("8")
                    quantile2=StringVar()
                    quantile2.set("10")
                    win2.destroy()
                    win3=Toplevel(main)
                    win3.title("Find peaks")
                    Label(win3,text="Select parameters to find peaks for condition "+cond2_name.get()).grid(row=0,column=0,padx=10,columnspan=3)
                    Label(win3,text="Frequency to start finding peaks (Hz):").grid(row=1,column=0,padx=10,sticky=W)
                    Entry(win3,textvariable=fpeakstart2,width=8).grid(row=1,column=1,sticky=W)
                    Label(win3,text="Sensitivity (0.0 to 10000):").grid(row=2,column=0,padx=10,sticky=W)
                    Entry(win3,textvariable=quantile2,width=8).grid(row=2,column=1,sticky=W)
                    def peakfind2():
                        psds, freqs = mne.time_frequency.psd_multitaper(raw2)
                        psds_mean=psds.mean(0)
                        f2=freqs[freqs>float(fpeakstart2.get())]
                        sig=psds_mean[np.where(freqs>float(fpeakstart2.get()))]
                        pk,prop=find_peaks(sig,distance=100,prominence=np.quantile(sig,1-float(quantile2.get())/10000))
                        wd,wdht,left_ips,right_ips=peak_widths(sig,pk)
                        sig2=10*np.log10(sig)
                        wdht2=10*np.log10(wdht)
                        plt.plot(f2,sig2)
                        plt.plot(f2[pk],sig2[pk],'x')
                        f_left=[f2[int(i)] for i in left_ips]
                        f_right=[f2[int(i)] for i in right_ips]
                        plt.hlines(wdht2,f_left,f_right,color="C2")
                        plt.xlabel("Frequency(Hz)")
                        plt.ylabel("PSD (dB)")
                        wd2=np.array([(f_right[i]-f_left[i])*2 for i in range(len(left_ips))])
                        peakbtn["text"]="Redo find peaks"
                        def deletepeaks2():
                            global raw2
                            if len(pk>0):
                                filt_raw=mne.filter.notch_filter(raw2.get_data(),raw2.info['sfreq'],f2[pk],fir_window='hann',notch_widths=wd2)
                                raw2=mne.io.RawArray(filt_raw,raw2.info)
                            win3.destroy()
                            plt.close("all")
                            showinfo(title="Info",message="Selected frequency peaks\ndeleted in condition\n"+cond2_name.get())
                            step3()
                        Button(win3,text="Peaks correctly found\nproceed with filter",command=deletepeaks2).grid(row=11,column=0,columnspan=3,pady=10,padx=10)
                        plt.show()
                    peakbtn=Button(win3,text="Find peaks",command=peakfind2)
                    peakbtn.grid(row=10,column=0,columnspan=3,pady=10,padx=10)
                if hasdbs.get()==4:
                    step3()
            Button(win2,text="OK",command=step2).grid(row=8,column=0,padx=10,columnspan=5)
        Button(win,text="OK",command=step1).grid(row=2,column=0,padx=10)
        raw1.plot_sensors(show_names=True)
        fig1=raw1.plot_psd_topo(fmin=1,fmax=50,color='k',fig_facecolor='w',axis_facecolor='w',show=False)
        fig1.suptitle(cond1_name.get())
        fig1.show()
        if raw2 is not None:
            fig2=raw2.plot_psd_topo(fmin=1,fmax=50,color='k',fig_facecolor='w',axis_facecolor='w',show=False)
            fig2.suptitle(cond2_name.get())
            fig2.show()
        win.lift()
        

#notch filter to select frequency band
def select_freq():
    global eeg1
    global eeg2
    if eeg1 is not None:
        st_or_freq=IntVar()
        st_or_freq.set(1)
        optionband=['Alpha (8-12 Hz)', 'Beta (12.5-30Hz)','Gamma (30-50Hz)','Theta (4-7Hz)','Delta (1-4Hz)',
                    'Alpha-1 (8-10Hz)','Alpha-2 (10.5-12.5Hz)','Beta-1 (12.5-16Hz)','Beta-2 (16.5-20Hz)','Beta-3 (20.5-28Hz)']
        sel_band=StringVar()
        sel_band.set(optionband[0])
        fmin=StringVar()
        fmin.set(8)
        fmax=StringVar()
        fmax.set(12)
        win=Toplevel(main)
        Label(win,text="Select frequency range:").grid(row=0,column=0,padx=10,pady=10,sticky=W)
        Radiobutton(win,text="From stardard",variable=st_or_freq,value=1).grid(row=1,column=0,padx=10,sticky=W)
        OptionMenu(win,sel_band,*optionband).grid(row=1,column=1,sticky=W)
        Radiobutton(win,text="Other:",variable=st_or_freq,value=2).grid(row=2,column=0,padx=10,sticky=W)
        Label(win,text="Minimum frequency (Hz):").grid(row=2,column=1,sticky=E)
        Entry(win,textvariable=fmin,width=4).grid(row=2,column=2,sticky=W)
        Label(win,text="Maximum frequency (Hz):").grid(row=2,column=3,sticky=E)
        Entry(win,textvariable=fmax,width=4).grid(row=2,column=4,sticky=W,padx=(0,10))
        def do_filter():
            global eeg1
            global eeg2
            if st_or_freq.get()==1:
                band_ranges=[(8,12),(12.5,30),(30,50),(4,7),(1,4),(8,10),(10.5,12.5),(12.5,16),(16.5,20),(20.5,28)]
                for i in range(len(optionband)):
                    if sel_band.get()==optionband[i]:
                        fmin.set(str(band_ranges[i][0]))
                        fmax.set(str(band_ranges[i][1]))
            eeg1=eeg1.copy().filter(float(fmin.get()),float(fmax.get()))
            if eeg2 is not None:
                eeg2=eeg2.copy().filter(float(fmin.get()),float(fmax.get()))
            win.destroy()
            showinfo(title="Filter applied",message="Filter applied to pre-processed EEG data.\nSelected frequency band range "+fmin.get()+" to "+fmax.get()+"Hz to remain")
        Button(win,text="OK",command=do_filter).grid(row=3,column=0,padx=10,columnspan=5)
    else:
        win=Toplevel(main)
        Label(win,text="ERROR\nIt is required at least\n1 preprocessed data",justify=CENTER).grid(row=0,column=0,padx=10,pady=10)        
        Button(win,text="OK",command=win.destroy).grid(row=1,column=0,padx=10)
    
#adjusts calcs if user selects to work with energy or raw values
def make_x():
    global x1
    global x2
    global xraw1
    global xraw2
    error=0
    if raw_or_energy.get()==1: #work with raw value
        #if epoched data is provided
        if eeg1 is not None:
            x1=eeg1.copy()
        else:
            error=1
        if eeg2 is not None:
            x2=eeg2.copy()
        #if continuous edf data is provided
        if raw1 is not None:
            xraw1=raw1.copy()
        if raw2 is not None:
            xraw2=raw2.copy()
    else: #work with squared value (energy)
        #if epoched data is provided
        if eeg1 is not None:
            a=np.square(eeg1.get_data())
            x1=mne.EpochsArray(a,eeg1.info,events=event_dict)
        else:
            error=1
        if eeg2 is not None:
            a=np.square(eeg2.get_data())
            x2=mne.EpochsArray(a,eeg2.info,events=event_dict)
        #if continuous edf data is provided
        if raw1 is not None:
            a=np.square(raw1.get_data())
            xraw1=mne.io.RawArray(a,raw1.info,events=event_dict)
        if raw2 is not None:
            a=np.square(raw2.get_data())
            xraw2=mne.io.RawArray(a,raw2.info,events=event_dict)        
    return error

#pearson correlation and comparison between conditions
def pearson_corr():
    win=Toplevel(main)
    global x1
    global x2
    delay=StringVar()
    delay.set("10")
    error=make_x()
    if error==1:
        Label(win,text="ERROR\nIt is required at least\n1 preprocessed data",justify=CENTER).grid(row=0,column=0,padx=10,pady=10)        
        Button(win,text="OK",command=win.destroy).grid(row=1,column=0,padx=10)
    else:
        n_chans=len(eeg1.ch_names)
        def step():
            if int(delay.get())>=0:
                delayval=int(float(delay.get())*eeg1.info['sfreq']/1000)
                max_epo_x1=max([len(x1[key]) for key in event_dict.keys()])
                corr_val_x1=np.empty((len(event_dict.keys()),max_epo_x1,n_chans,n_chans))
                corr_val_x1.fill(np.nan)
                if x2 is not None:
                    max_epo_x2=max([len(x2[key]) for key in event_dict.keys()])
                    corr_val_x2=np.empty((len(event_dict.keys()),max_epo_x2,n_chans,n_chans))
                    corr_val_x2.fill(np.nan)
                pbar['value']=0.0
                k=0                
                for i in range(n_chans):
                    for j in range(n_chans):
                        win.update_idletasks()
                        win.update()
                        k+=1
                        pbar['value'] += 100/(n_chans*n_chans)
                        pbtxt['text']=f"{k:d}/{n_chans*n_chans:d}"
                        key_idx=0
                        for key in event_dict.keys():
                            n_epochs=len(x1[key])
                            for epoch in range(n_epochs):
                                if delayval>0:
                                    corr_val_x1[key_idx,epoch,i,j]=np.corrcoef(x1[key][epoch]._data[0,i,:-delayval],x1[key][epoch]._data[0,j,delayval:])[0,1]
                                elif delayval==0:
                                    corr_val_x1[key_idx,epoch,i,j]=np.corrcoef(x1[key][epoch]._data[0,i,:],x1[key][epoch]._data[0,j,:])[0,1]
                                else:
                                    showinfo(title="Error",message="Delay must be positive")
                            if x2 is not None:
                                n_epochs2=len(x2[key])
                                for epoch2 in range(n_epochs2):
                                    if delayval>0:
                                        corr_val_x2[key_idx,epoch2,i,j]=np.corrcoef(x2[key][epoch2]._data[0,i,:-delayval],x2[key][epoch2]._data[0,j,delayval:])[0,1]
                                    elif delayval==0:
                                        corr_val_x2[key_idx,epoch2,i,j]=np.corrcoef(x2[key][epoch2]._data[0,i,:],x2[key][epoch2]._data[0,j,:])[0,1]
                                    else:
                                        showinfo(title="Error",message="Delay must be positive")
                            key_idx+=1
                def save_corr():
                    showinfo(title="Info",message="First you'll save the average\nacross epochs (CSV table)")
                    key_idx=0
                    for key in event_dict.keys():
                        #calculate mean correlations
                        mean_corr1=np.empty((n_chans,n_chans))
                        mean_corr1.fill(np.nan)
                        if x2 is not None:
                            mean_corr2=np.empty((n_chans,n_chans))
                            mean_corr2.fill(np.nan)
                        for i in range(n_chans):
                            for j in range(n_chans):
                                mean_corr1[i,j]=np.nanmean(corr_val_x1[key_idx,:,i,j])
                                if x2 is not None:
                                    mean_corr2[i,j]=np.nanmean(corr_val_x2[key_idx,:,i,j])
                        #save
                        fname1 = fd.asksaveasfilename(title="Correlation condition "+cond1_name.get()+" event "+key,defaultextension=".csv",filetypes=(("Comma Separated Values", "*.csv"),("All Files", "*.*")))
                        df=pd.DataFrame(mean_corr1,index=x1.ch_names,columns=x1.ch_names)
                        df.to_csv(fname1)
                        if x2 is not None:
                            fname2 = fd.asksaveasfilename(title="Correlation condition "+cond2_name.get()+" event "+key,defaultextension=".csv",filetypes=(("Comma Separated Values", "*.csv"),("All Files", "*.*")))
                            df2=pd.DataFrame(mean_corr2,index=x2.ch_names,columns=x2.ch_names)
                            df2.to_csv(fname2)
                        key_idx+=1
                    #save np.array
                    showinfo(title="Info",message="Now you'll save the full results\n(Numpy array of size (n_events,n_epochs,n_chans,n_chans))")
                    fname1 = fd.asksaveasfilename(title="Correlation condition "+cond1_name.get(),defaultextension=".npy",filetypes=(("Numpy array", "*.npy"),("All Files", "*.*")))
                    np.save(fname1,corr_val_x1)
                    if x2 is not None:
                        fname2 = fd.asksaveasfilename(title="Correlation condition "+cond2_name.get(),defaultextension=".npy",filetypes=(("Numpy array", "*.npy"),("All Files", "*.*")))
                        np.save(fname2,corr_val_x2)
                def plot_corr():
                    #calculate mean correlations
                    key_idx=0
                    mean_corr1=np.empty((len(event_dict.keys()),n_chans,n_chans))
                    mean_corr1.fill(np.nan)
                    if x2 is not None:
                        mean_corr2=np.empty((len(event_dict.keys()),n_chans,n_chans))
                        mean_corr2.fill(np.nan)
                    for key in event_dict.keys():
                        for i in range(n_chans):
                            for j in range(n_chans):
                                mean_corr1[key_idx,i,j]=np.nanmean(corr_val_x1[key_idx,:,i,j])
                                if x2 is not None:
                                    mean_corr2[key_idx,i,j]=np.nanmean(corr_val_x2[key_idx,:,i,j])
                        key_idx+=1
                    #plot
                    if x2 is None:
                        if len(event_dict.keys())==1:
                            fig, axes = plt.subplots(nrows=1, ncols=1)
                            df=pd.DataFrame(mean_corr1[0,:,:],index=x1.ch_names,columns=x1.ch_names)
                            sns.heatmap(df,xticklabels=df.index,yticklabels=df.columns,mask=df.isnull(),cmap='bwr',ax=axes,vmin=-1,vmax=1)
                            plt.xlabel("Electrodes")
                            plt.ylabel("Electrodes")
                            axes.figure.axes[-1].set_ylabel('Pearson correlation\n(average across epochs)',size=14)
                            plt.show()
                        elif len(event_dict.keys())>2:
                            fig,axes = plt.subplots(nrows=len(event_dict.keys()))
                            dfs=[]
                            rows=list(event_dict.keys())
                            for i in range(len(event_dict.keys())):
                                dfs.append(pd.DataFrame(mean_corr1[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                            pad=5
                            for ax, row in zip(axes[:], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for i in range(len(event_dict.keys())):
                                sns.heatmap(dfs[i],xticklabels=dfs[i].index,yticklabels=dfs[i].columns,mask=dfs[i].isnull(),cmap='bwr',ax=axes[i],vmin=-1,vmax=1)
                                axes[i].set_ylabel('Electrodes')
                                axes[i].set_xlabel('Electrodes')
                                axes[i].figure.axes[-1].set_ylabel('Pearson correlation\n(average across epochs)')
                            plt.show()
                        elif len(event_dict.keys())==2:
                            fig,axes = plt.subplots(nrows=3)
                            dfs=[]
                            rows=list(event_dict.keys())+['Difference\n'+list(event_dict.keys())[1]+'\n'+list(event_dict.keys())[0]]
                            for i in range(2):
                                dfs.append(pd.DataFrame(mean_corr1[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                            dfs.append(pd.DataFrame(dfs[1]-dfs[0]))
                            pad=5
                            for ax, row in zip(axes[:], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for i in range(len(rows)):
                                sns.heatmap(dfs[i],xticklabels=dfs[i].index,yticklabels=dfs[i].columns,mask=dfs[i].isnull(),cmap='bwr',ax=axes[i],vmin=-1,vmax=1)
                                axes[i].set_ylabel('Electrodes')
                                axes[i].set_xlabel('Electrodes')
                                if i==len(rows)-1:
                                    axes[i].figure.axes[-1].set_ylabel('Difference of\nPearson correlation values')
                                else:
                                    axes[i].figure.axes[-1].set_ylabel('Pearson correlation\n(average across epochs)')
                            plt.show()
                    else:
                        if len(event_dict.keys())==1:
                            fig, axes = plt.subplots(nrows=1, ncols=3)
                            df1=pd.DataFrame(mean_corr1[0,:,:],index=x1.ch_names,columns=x1.ch_names)
                            df2=pd.DataFrame(mean_corr2[0,:,:],index=x2.ch_names,columns=x2.ch_names)
                            df_diff=pd.DataFrame(df2-df1)
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            pad=5
                            for ax, col in zip(axes[:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            sns.heatmap(df1,xticklabels=df1.index,yticklabels=df1.columns,mask=df1.isnull(),cmap='bwr',ax=axes[0],vmin=-1,vmax=1)
                            axes[0].set_ylabel('Electrodes')
                            axes[0].set_xlabel('Electrodes')
                            axes[0].figure.axes[-1].set_ylabel('Pearson correlation\n(average across epochs)')
                            sns.heatmap(df2,xticklabels=df2.index,yticklabels=df2.columns,mask=df2.isnull(),cmap='bwr',ax=axes[1],vmin=-1,vmax=1)
                            axes[1].set_ylabel('Electrodes')
                            axes[1].set_xlabel('Electrodes')
                            axes[1].figure.axes[-1].set_ylabel('Pearson correlation\n(average across epochs)')
                            sns.heatmap(df_diff,xticklabels=df_diff.index,yticklabels=df_diff.columns,mask=df_diff.isnull(),cmap='bwr',ax=axes[2],vmin=-1,vmax=1)
                            axes[2].set_ylabel('Electrodes')
                            axes[2].set_xlabel('Electrodes')
                            axes[2].figure.axes[-1].set_ylabel('Difference of\nPearson correlation values')
                            plt.show()
                        elif len(event_dict.keys())>2:
                            fig,axes = plt.subplots(nrows=len(event_dict.keys()),ncols=3)
                            dfs1=[]
                            dfs2=[]
                            dfs_diff=[]
                            rows=list(event_dict.keys())
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            for i in range(len(event_dict.keys())):
                                dfs1.append(pd.DataFrame(mean_corr1[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                                dfs2.append(pd.DataFrame(mean_corr2[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                                dfs_diff.append(pd.DataFrame(dfs2[i]-dfs1[i]))
                            pad=5
                            for ax, row in zip(axes[:,0], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',size=17,ha='right',va='center')
                            for ax, col in zip(axes[0,:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            for i in range(len(event_dict.keys())):
                                sns.heatmap(dfs1[i],xticklabels=dfs1[i].index,yticklabels=dfs1[i].columns,mask=dfs1[i].isnull(),cmap='bwr',ax=axes[i,0],vmin=-1,vmax=1)
                                axes[i,0].set_ylabel('Electrodes')
                                axes[i,0].set_xlabel('Electrodes')
                                axes[i,0].figure.axes[-1].set_ylabel('Pearson correlation\n(average across epochs)')
                                sns.heatmap(df2,xticklabels=dfs2[i].index,yticklabels=dfs2[i].columns,mask=dfs2[i].isnull(),cmap='bwr',ax=axes[i,1],vmin=-1,vmax=1)
                                axes[i,1].set_ylabel('Electrodes')
                                axes[i,1].set_xlabel('Electrodes')
                                axes[i,1].figure.axes[-1].set_ylabel('Pearson correlation\n(average across epochs)')
                                sns.heatmap(dfs_diff[i],xticklabels=dfs_diff[i].index,yticklabels=dfs_diff[i].columns,mask=dfs_diff[i].isnull(),cmap='bwr',ax=axes[i,2],vmin=-1,vmax=1)
                                axes[i,2].set_ylabel('Electrodes')
                                axes[i,2].set_xlabel('Electrodes')
                                axes[i,2].figure.axes[-1].set_ylabel('Difference of\nPearson correlation values')
                            plt.show()
                        elif len(event_dict.keys())==2:
                            fig,axes = plt.subplots(nrows=3,ncols=3)
                            dfs1=[]
                            dfs2=[]
                            dfs_diff=[]
                            rows=list(event_dict.keys())+['Difference\n'+list(event_dict.keys())[1]+'\n'+list(event_dict.keys())[0]]
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            for i in range(2):
                                dfs1.append(pd.DataFrame(mean_corr1[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                                dfs2.append(pd.DataFrame(mean_corr2[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                                dfs_diff.append(pd.DataFrame(dfs2[i]-dfs1[i]))
                            dfs1.append(pd.DataFrame(dfs1[1]-dfs1[0]))
                            dfs2.append(pd.DataFrame(dfs2[1]-dfs2[0]))
                            pad=5
                            for ax, row in zip(axes[:,0], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for ax, col in zip(axes[0,:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            for i in range(3):
                                sns.heatmap(dfs1[i],xticklabels=dfs1[i].index,yticklabels=dfs1[i].columns,mask=dfs1[i].isnull(),cmap='bwr',ax=axes[i,0],vmin=-1,vmax=1)
                                axes[i,0].set_ylabel('Electrodes')
                                axes[i,0].set_xlabel('Electrodes')
                                if i<2:
                                    axes[i,0].figure.axes[-1].set_ylabel('Pearson correlation\n(average across epochs)')
                                else:
                                    axes[i,0].figure.axes[-1].set_ylabel('Difference of\nPearson correlation values')
                                sns.heatmap(dfs2[i],xticklabels=dfs2[i].index,yticklabels=dfs2[i].columns,mask=dfs2[i].isnull(),cmap='bwr',ax=axes[i,1],vmin=-1,vmax=1)
                                axes[i,1].set_ylabel('Electrodes')
                                axes[i,1].set_xlabel('Electrodes')
                                if i<2:
                                    axes[i,1].figure.axes[-1].set_ylabel('Pearson correlation\n(average across epochs)')
                                else:
                                    axes[i,1].figure.axes[-1].set_ylabel('Difference of\nPearson correlation values')
                                if i<2:
                                    sns.heatmap(dfs_diff[i],xticklabels=dfs_diff[i].index,yticklabels=dfs_diff[i].columns,mask=dfs_diff[i].isnull(),cmap='bwr',ax=axes[i,2],vmin=-1,vmax=1)
                                    axes[i,2].set_ylabel('Electrodes')
                                    axes[i,2].set_xlabel('Electrodes')
                                    axes[i,2].figure.axes[-1].set_ylabel('Difference of\nPearson correlation values')
                            fig.delaxes(axes[2,2])
                            plt.show()
                Button(win,text="Save results",command=save_corr).grid(row=3,column=0,padx=10,sticky=W)
                Button(win,text="Make plot",command=plot_corr).grid(row=4,column=0,padx=10,sticky=W)
                Button(win,text="Close",command=win.destroy).grid(row=5,column=0,padx=10,pady=10,columnspan=3)
            else:
                showinfo(title="Error",message="Delay must be a positive number")
        Label(win,text="Calculate correlations").grid(row=0,column=0,columnspan=3,padx=10,pady=10)
        Label(win,text="Transmission delay between brain regions (ms):").grid(row=1,column=0,padx=10,sticky=W)
        Entry(win,textvariable=delay,width=4).grid(row=1,column=1,sticky=W)
        pbar=Progressbar(win,orient=HORIZONTAL,length=100,mode='determinate')
        pbar['value']=0.0
        pbtxt=Label(win,text="--")
        btn=Button(win, text='Start calculation', command=step)
        btn.grid(row=2,column=0,sticky=W,padx=10)
        pbar.grid(row=2,column=1,sticky=W)
        pbtxt.grid(row=2,column=2,sticky=W)        

#spearman correlation and comparison between conditions
def spearman_corr():
    win=Toplevel(main)
    global x1
    global x2
    delay=StringVar()
    delay.set("10")
    error=make_x()
    if error==1:
        Label(win,text="ERROR\nIt is required at least\n1 preprocessed data",justify=CENTER).grid(row=0,column=0,padx=10,pady=10)        
        Button(win,text="OK",command=win.destroy).grid(row=1,column=0,padx=10)
    else:
        n_chans=len(eeg1.ch_names)
        def step():
            if int(delay.get())>=0:
                delayval=int(float(delay.get())*eeg1.info['sfreq']/1000)
                max_epo_x1=max([len(x1[key]) for key in event_dict.keys()])
                corr_val_x1=np.empty((len(event_dict.keys()),max_epo_x1,n_chans,n_chans))
                corr_val_x1.fill(np.nan)
                if x2 is not None:
                    max_epo_x2=max([len(x2[key]) for key in event_dict.keys()])
                    corr_val_x2=np.empty((len(event_dict.keys()),max_epo_x2,n_chans,n_chans))
                    corr_val_x2.fill(np.nan)
                pbar['value']=0.0
                k=0                
                for i in range(n_chans):
                    for j in range(n_chans):
                        win.update_idletasks()
                        win.update()
                        k+=1
                        pbar['value'] += 100/(n_chans*n_chans)
                        pbtxt['text']=f"{k:d}/{n_chans*n_chans:d}"
                        key_idx=0
                        for key in event_dict.keys():
                            n_epochs=len(x1[key])
                            for epoch in range(n_epochs):
                                if delayval>0:
                                    corr_val_x1[key_idx,epoch,i,j]=spearmanr(x1[key][epoch]._data[0,i,:-delayval],x1[key][epoch]._data[0,j,delayval:]).correlation
                                elif delayval==0:
                                    corr_val_x1[key_idx,epoch,i,j]=spearmanr(x1[key][epoch]._data[0,i,:],x1[key][epoch]._data[0,j,:]).correlation
                                else:
                                    showinfo(title="Error",message="Delay must be positive")
                            if x2 is not None:
                                n_epochs2=len(x2[key])
                                for epoch2 in range(n_epochs2):
                                    if delayval>0:
                                        corr_val_x2[key_idx,epoch2,i,j]=spearmanr(x2[key][epoch2]._data[0,i,:-delayval],x2[key][epoch2]._data[0,j,delayval:]).correlation
                                    elif delayval==0:
                                        corr_val_x2[key_idx,epoch2,i,j]=spearmanr(x2[key][epoch2]._data[0,i,:],x2[key][epoch2]._data[0,j,:]).correlation
                                    else:
                                        showinfo(title="Error",message="Delay must be positive")
                            key_idx+=1
                def save_corr():
                    showinfo(title="Info",message="First you'll save the average\nacross epochs (CSV table)")
                    key_idx=0
                    for key in event_dict.keys():
                        #calculate mean correlations
                        mean_corr1=np.empty((n_chans,n_chans))
                        mean_corr1.fill(np.nan)
                        if x2 is not None:
                            mean_corr2=np.empty((n_chans,n_chans))
                            mean_corr2.fill(np.nan)
                        for i in range(n_chans):
                            for j in range(n_chans):
                                mean_corr1[i,j]=np.nanmean(corr_val_x1[key_idx,:,i,j])
                                if x2 is not None:
                                    mean_corr2[i,j]=np.nanmean(corr_val_x2[key_idx,:,i,j])
                        #save
                        fname1 = fd.asksaveasfilename(title="Correlation condition "+cond1_name.get()+" event "+key,defaultextension=".csv",filetypes=(("Comma Separated Values", "*.csv"),("All Files", "*.*")))
                        df=pd.DataFrame(mean_corr1,index=x1.ch_names,columns=x1.ch_names)
                        df.to_csv(fname1)
                        if x2 is not None:
                            fname2 = fd.asksaveasfilename(title="Correlation condition "+cond2_name.get()+" event "+key,defaultextension=".csv",filetypes=(("Comma Separated Values", "*.csv"),("All Files", "*.*")))
                            df2=pd.DataFrame(mean_corr2,index=x2.ch_names,columns=x2.ch_names)
                            df2.to_csv(fname2)
                        key_idx+=1
                    #save np.array
                    showinfo(title="Info",message="Now you'll save the full results\n(Numpy array of size (n_events,n_epochs,n_chans,n_chans))")
                    fname1 = fd.asksaveasfilename(title="Correlation condition "+cond1_name.get(),defaultextension=".npy",filetypes=(("Numpy array", "*.npy"),("All Files", "*.*")))
                    np.save(fname1,corr_val_x1)
                    if x2 is not None:
                        fname2 = fd.asksaveasfilename(title="Correlation condition "+cond2_name.get(),defaultextension=".npy",filetypes=(("Numpy array", "*.npy"),("All Files", "*.*")))
                        np.save(fname2,corr_val_x2)
                def plot_corr():
                    #calculate mean correlations
                    key_idx=0
                    mean_corr1=np.empty((len(event_dict.keys()),n_chans,n_chans))
                    mean_corr1.fill(np.nan)
                    if x2 is not None:
                        mean_corr2=np.empty((len(event_dict.keys()),n_chans,n_chans))
                        mean_corr2.fill(np.nan)
                    for key in event_dict.keys():
                        for i in range(n_chans):
                            for j in range(n_chans):
                                mean_corr1[key_idx,i,j]=np.nanmean(corr_val_x1[key_idx,:,i,j])
                                if x2 is not None:
                                    mean_corr2[key_idx,i,j]=np.nanmean(corr_val_x2[key_idx,:,i,j])
                        key_idx+=1
                    #plot
                    if x2 is None:
                        if len(event_dict.keys())==1:
                            fig, axes = plt.subplots(nrows=1, ncols=1)
                            df=pd.DataFrame(mean_corr1[0,:,:],index=x1.ch_names,columns=x1.ch_names)
                            sns.heatmap(df,xticklabels=df.index,yticklabels=df.columns,mask=df.isnull(),cmap='bwr',ax=axes,vmin=-1,vmax=1)
                            plt.xlabel("Electrodes")
                            plt.ylabel("Electrodes")
                            axes.figure.axes[-1].set_ylabel('Spearman correlation\n(average across epochs)',size=14)
                            plt.show()
                        elif len(event_dict.keys())>2:
                            fig,axes = plt.subplots(nrows=len(event_dict.keys()))
                            dfs=[]
                            rows=list(event_dict.keys())
                            for i in range(len(event_dict.keys())):
                                dfs.append(pd.DataFrame(mean_corr1[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                            pad=5
                            for ax, row in zip(axes[:], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for i in range(len(event_dict.keys())):
                                sns.heatmap(dfs[i],xticklabels=dfs[i].index,yticklabels=dfs[i].columns,mask=dfs[i].isnull(),cmap='bwr',ax=axes[i],vmin=-1,vmax=1)
                                axes[i].set_ylabel('Electrodes')
                                axes[i].set_xlabel('Electrodes')
                                axes[i].figure.axes[-1].set_ylabel('Spearman correlation\n(average across epochs)')
                            plt.show()
                        elif len(event_dict.keys())==2:
                            fig,axes = plt.subplots(nrows=3)
                            dfs=[]
                            rows=list(event_dict.keys())+['Difference\n'+list(event_dict.keys())[1]+'\n'+list(event_dict.keys())[0]]
                            for i in range(2):
                                dfs.append(pd.DataFrame(mean_corr1[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                            dfs.append(pd.DataFrame(dfs[1]-dfs[0]))
                            pad=5
                            for ax, row in zip(axes[:], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for i in range(len(rows)):
                                sns.heatmap(dfs[i],xticklabels=dfs[i].index,yticklabels=dfs[i].columns,mask=dfs[i].isnull(),cmap='bwr',ax=axes[i],vmin=-1,vmax=1)
                                axes[i].set_ylabel('Electrodes')
                                axes[i].set_xlabel('Electrodes')
                                if i==len(rows)-1:
                                    axes[i].figure.axes[-1].set_ylabel('Difference of\nSpearman correlation values')
                                else:
                                    axes[i].figure.axes[-1].set_ylabel('Spearman correlation\n(average across epochs)')
                            plt.show()
                    else:
                        if len(event_dict.keys())==1:
                            fig, axes = plt.subplots(nrows=1, ncols=3)
                            df1=pd.DataFrame(mean_corr1[0,:,:],index=x1.ch_names,columns=x1.ch_names)
                            df2=pd.DataFrame(mean_corr2[0,:,:],index=x2.ch_names,columns=x2.ch_names)
                            df_diff=pd.DataFrame(df2-df1)
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            pad=5
                            for ax, col in zip(axes[:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            sns.heatmap(df1,xticklabels=df1.index,yticklabels=df1.columns,mask=df1.isnull(),cmap='bwr',ax=axes[0],vmin=-1,vmax=1)
                            axes[0].set_ylabel('Electrodes')
                            axes[0].set_xlabel('Electrodes')
                            axes[0].figure.axes[-1].set_ylabel('Spearman correlation\n(average across epochs)')
                            sns.heatmap(df2,xticklabels=df2.index,yticklabels=df2.columns,mask=df2.isnull(),cmap='bwr',ax=axes[1],vmin=-1,vmax=1)
                            axes[1].set_ylabel('Electrodes')
                            axes[1].set_xlabel('Electrodes')
                            axes[1].figure.axes[-1].set_ylabel('Spearman correlation\n(average across epochs)')
                            sns.heatmap(df_diff,xticklabels=df_diff.index,yticklabels=df_diff.columns,mask=df_diff.isnull(),cmap='bwr',ax=axes[2],vmin=-1,vmax=1)
                            axes[2].set_ylabel('Electrodes')
                            axes[2].set_xlabel('Electrodes')
                            axes[2].figure.axes[-1].set_ylabel('Difference of\nSpearman correlation values')
                            plt.show()
                        elif len(event_dict.keys())>2:
                            fig,axes = plt.subplots(nrows=len(event_dict.keys()),ncols=3)
                            dfs1=[]
                            dfs2=[]
                            dfs_diff=[]
                            rows=list(event_dict.keys())
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            for i in range(len(event_dict.keys())):
                                dfs1.append(pd.DataFrame(mean_corr1[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                                dfs2.append(pd.DataFrame(mean_corr2[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                                dfs_diff.append(pd.DataFrame(dfs2[i]-dfs1[i]))
                            pad=5
                            for ax, row in zip(axes[:,0], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',size=17,ha='right',va='center')
                            for ax, col in zip(axes[0,:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            for i in range(len(event_dict.keys())):
                                sns.heatmap(dfs1[i],xticklabels=dfs1[i].index,yticklabels=dfs1[i].columns,mask=dfs1[i].isnull(),cmap='bwr',ax=axes[i,0],vmin=-1,vmax=1)
                                axes[i,0].set_ylabel('Electrodes')
                                axes[i,0].set_xlabel('Electrodes')
                                axes[i,0].figure.axes[-1].set_ylabel('Spearman correlation\n(average across epochs)')
                                sns.heatmap(df2,xticklabels=dfs2[i].index,yticklabels=dfs2[i].columns,mask=dfs2[i].isnull(),cmap='bwr',ax=axes[i,1],vmin=-1,vmax=1)
                                axes[i,1].set_ylabel('Electrodes')
                                axes[i,1].set_xlabel('Electrodes')
                                axes[i,1].figure.axes[-1].set_ylabel('Spearman correlation\n(average across epochs)')
                                sns.heatmap(dfs_diff[i],xticklabels=dfs_diff[i].index,yticklabels=dfs_diff[i].columns,mask=dfs_diff[i].isnull(),cmap='bwr',ax=axes[i,2],vmin=-1,vmax=1)
                                axes[i,2].set_ylabel('Electrodes')
                                axes[i,2].set_xlabel('Electrodes')
                                axes[i,2].figure.axes[-1].set_ylabel('Difference of\nSpearman correlation values')
                            plt.show()
                        elif len(event_dict.keys())==2:
                            fig,axes = plt.subplots(nrows=3,ncols=3)
                            dfs1=[]
                            dfs2=[]
                            dfs_diff=[]
                            rows=list(event_dict.keys())+['Difference\n'+list(event_dict.keys())[1]+'\n'+list(event_dict.keys())[0]]
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            for i in range(2):
                                dfs1.append(pd.DataFrame(mean_corr1[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                                dfs2.append(pd.DataFrame(mean_corr2[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                                dfs_diff.append(pd.DataFrame(dfs2[i]-dfs1[i]))
                            dfs1.append(pd.DataFrame(dfs1[1]-dfs1[0]))
                            dfs2.append(pd.DataFrame(dfs2[1]-dfs2[0]))
                            pad=5
                            for ax, row in zip(axes[:,0], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for ax, col in zip(axes[0,:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            for i in range(3):
                                sns.heatmap(dfs1[i],xticklabels=dfs1[i].index,yticklabels=dfs1[i].columns,mask=dfs1[i].isnull(),cmap='bwr',ax=axes[i,0],vmin=-1,vmax=1)
                                axes[i,0].set_ylabel('Electrodes')
                                axes[i,0].set_xlabel('Electrodes')
                                if i<2:
                                    axes[i,0].figure.axes[-1].set_ylabel('Spearman correlation\n(average across epochs)')
                                else:
                                    axes[i,0].figure.axes[-1].set_ylabel('Difference of\nSpearman correlation values')
                                sns.heatmap(dfs2[i],xticklabels=dfs2[i].index,yticklabels=dfs2[i].columns,mask=dfs2[i].isnull(),cmap='bwr',ax=axes[i,1],vmin=-1,vmax=1)
                                axes[i,1].set_ylabel('Electrodes')
                                axes[i,1].set_xlabel('Electrodes')
                                if i<2:
                                    axes[i,1].figure.axes[-1].set_ylabel('Spearman correlation\n(average across epochs)')
                                else:
                                    axes[i,1].figure.axes[-1].set_ylabel('Difference of\nSpearman correlation values')
                                if i<2:
                                    sns.heatmap(dfs_diff[i],xticklabels=dfs_diff[i].index,yticklabels=dfs_diff[i].columns,mask=dfs_diff[i].isnull(),cmap='bwr',ax=axes[i,2],vmin=-1,vmax=1)
                                    axes[i,2].set_ylabel('Electrodes')
                                    axes[i,2].set_xlabel('Electrodes')
                                    axes[i,2].figure.axes[-1].set_ylabel('Difference of\nSpearman correlation values')
                            fig.delaxes(axes[2,2])
                            plt.show()
                Button(win,text="Save results",command=save_corr).grid(row=3,column=0,padx=10,sticky=W)
                Button(win,text="Make plot",command=plot_corr).grid(row=4,column=0,padx=10,sticky=W)
                Button(win,text="Close",command=win.destroy).grid(row=5,column=0,padx=10,pady=10,columnspan=3)
            else:
                showinfo(title="Error",message="Delay must be a positive number")
        Label(win,text="Calculate correlations").grid(row=0,column=0,columnspan=3,padx=10,pady=10)
        Label(win,text="Transmission delay between brain regions (ms):").grid(row=1,column=0,padx=10,sticky=W)
        Entry(win,textvariable=delay,width=4).grid(row=1,column=1,sticky=W)
        pbar=Progressbar(win,orient=HORIZONTAL,length=100,mode='determinate')
        pbar['value']=0.0
        pbtxt=Label(win,text="--")
        btn=Button(win, text='Start calculation', command=step)
        btn.grid(row=2,column=0,sticky=W,padx=10)
        pbar.grid(row=2,column=1,sticky=W)
        pbtxt.grid(row=2,column=2,sticky=W)

#transfer entropy and comparison between conditions
def te():
    win=Toplevel(main)
    global x1
    global x2
    delay=StringVar()
    delay.set("10")
    ns=StringVar()
    ns.set('5')
    div_type=IntVar()
    div_type.set(3)
    xdiv_vals=StringVar()
    xdiv_vals.set('-30,-10,10,30')
    ydiv_vals=StringVar()
    ydiv_vals.set('-30,-10,10,30')
    lxp=StringVar()
    lxp.set('1')
    lxp=StringVar()
    lxp.set('1')
    lyp=StringVar()
    lyp.set('1')
    lyf=StringVar()
    lyf.set('1')
    tau=StringVar()
    tau.set('1')
    unit=StringVar()
    optionlist=['bits','nat','ban']
    error=make_x()
    if error==1:
        Label(win,text="ERROR\nIt is required at least\n1 preprocessed data",justify=CENTER).grid(row=0,column=0,padx=10,pady=10)        
        Button(win,text="OK",command=win.destroy).grid(row=1,column=0,padx=10)
    else:
        n_chans=len(eeg1.ch_names)
        def step():
            if int(delay.get())>=0:
                delayval=int(float(delay.get())*eeg1.info['sfreq']/1000)
                max_epo_x1=max([len(x1[key]) for key in event_dict.keys()])
                corr_val_x1=np.empty((len(event_dict.keys()),max_epo_x1,n_chans,n_chans))
                corr_val_x1.fill(np.nan)
                if x2 is not None:
                    max_epo_x2=max([len(x2[key]) for key in event_dict.keys()])
                    corr_val_x2=np.empty((len(event_dict.keys()),max_epo_x2,n_chans,n_chans))
                    corr_val_x2.fill(np.nan)
                if div_type.get()==1:
                    symb_type='equal-divs'
                    x_divs,y_divs=None,None
                elif div_type.get()==2:
                    symb_type='equal-points'
                    x_divs,y_divs=None,None
                elif div_type.get()==3:
                    x_divs=[float(i) for i in xdiv_vals.get().split(sep=',')]
                    y_divs=[float(i) for i in ydiv_vals.get().split(sep=',')]
                    symb_type=None
                total_steps=0
                for i in range(n_chans):
                    for j in range(n_chans):
                        for key in event_dict.keys():
                            for epoch1len in range(len(x1[key])):
                                total_steps+=1
                            if x2 is not None:
                                for epoch2len in range(len(x2[key])):
                                    total_steps+=1            
                pbar['value']=0.0
                k=0                
                for i in range(n_chans):
                    for j in range(n_chans):
                        key_idx=0
                        for key in event_dict.keys():
                            n_epochs=len(x1[key])
                            for epoch in range(n_epochs):
                                win.update_idletasks()
                                win.update()
                                k+=1
                                pbar['value'] += 100/total_steps
                                pbtxt['text']=f"{k:d}/{total_steps:d}"
                                if delayval>0:
                                    corr_val_x1[key_idx,epoch,i,j]=transfer_entropy(x1[key][epoch]._data[0,i,:-delayval],x1[key][epoch]._data[0,j,delayval:],
                                                                                    symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                    symbolic_length=(int(lxp.get()),int(lyp.get()),int(lyf.get())),
                                                                                    x_divs=x_divs,y_divs=y_divs,units=unit.get())
                                elif delayval==0:
                                    corr_val_x1[key_idx,epoch,i,j]=transfer_entropy(x1[key][epoch]._data[0,i,:],x1[key][epoch]._data[0,j,:],
                                                                                    symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                    symbolic_length=(int(lxp.get()),int(lyp.get()),int(lyf.get())),
                                                                                    x_divs=x_divs,y_divs=y_divs,units=unit.get())
                                else:
                                    showinfo(title="Error",message="Delay must be positive")
                            if x2 is not None:
                                n_epochs2=len(x2[key])
                                for epoch2 in range(n_epochs2):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar['value'] += 100/total_steps
                                    pbtxt['text']=f"{k:d}/{total_steps:d}"
                                    if delayval>0:
                                        corr_val_x2[key_idx,epoch2,i,j]=transfer_entropy(x2[key][epoch2]._data[0,i,:-delayval],x2[key][epoch2]._data[0,j,delayval:],
                                                                                          symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                          symbolic_length=(int(lxp.get()),int(lyp.get()),int(lyf.get())),
                                                                                          x_divs=x_divs,y_divs=y_divs,units=unit.get())
                                    elif delayval==0:
                                        corr_val_x2[key_idx,epoch2,i,j]=transfer_entropy(x2[key][epoch2]._data[0,i,:],x2[key][epoch2]._data[0,j,:],
                                                                                          symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                          symbolic_length=(int(lxp.get()),int(lyp.get()),int(lyf.get())),
                                                                                          x_divs=x_divs,y_divs=y_divs,units=unit.get())
                                    else:
                                        showinfo(title="Error",message="Delay must be positive")
                            key_idx+=1
                def save_corr():
                    showinfo(title="Info",message="First you'll save the average\nacross epochs (CSV table)")
                    key_idx=0
                    for key in event_dict.keys():
                        #calculate mean correlations
                        mean_corr1=np.empty((n_chans,n_chans))
                        mean_corr1.fill(np.nan)
                        if x2 is not None:
                            mean_corr2=np.empty((n_chans,n_chans))
                            mean_corr2.fill(np.nan)
                        for i in range(n_chans):
                            for j in range(n_chans):
                                mean_corr1[i,j]=np.nanmean(corr_val_x1[key_idx,:,i,j])
                                if x2 is not None:
                                    mean_corr2[i,j]=np.nanmean(corr_val_x2[key_idx,:,i,j])
                        #save
                        fname1 = fd.asksaveasfilename(title="TE condition "+cond1_name.get()+" event "+key,defaultextension=".csv",filetypes=(("Comma Separated Values", "*.csv"),("All Files", "*.*")))
                        df=pd.DataFrame(mean_corr1,index=x1.ch_names,columns=x1.ch_names)
                        df.to_csv(fname1)
                        if x2 is not None:
                            fname2 = fd.asksaveasfilename(title="TE condition "+cond2_name.get()+" event "+key,defaultextension=".csv",filetypes=(("Comma Separated Values", "*.csv"),("All Files", "*.*")))
                            df2=pd.DataFrame(mean_corr2,index=x2.ch_names,columns=x2.ch_names)
                            df2.to_csv(fname2)
                        key_idx+=1
                    #save np.array
                    showinfo(title="Info",message="Now you'll save the full results\n(Numpy array of size (n_events,n_epochs,n_chans,n_chans))")
                    fname1 = fd.asksaveasfilename(title="TE condition "+cond1_name.get(),defaultextension=".npy",filetypes=(("Numpy array", "*.npy"),("All Files", "*.*")))
                    np.save(fname1,corr_val_x1)
                    if x2 is not None:
                        fname2 = fd.asksaveasfilename(title="TE condition "+cond2_name.get(),defaultextension=".npy",filetypes=(("Numpy array", "*.npy"),("All Files", "*.*")))
                        np.save(fname2,corr_val_x2)
                def plot_corr():
                    #calculate mean correlations
                    key_idx=0
                    mean_corr1=np.empty((len(event_dict.keys()),n_chans,n_chans))
                    mean_corr1.fill(np.nan)
                    if x2 is not None:
                        mean_corr2=np.empty((len(event_dict.keys()),n_chans,n_chans))
                        mean_corr2.fill(np.nan)
                    for key in event_dict.keys():
                        for i in range(n_chans):
                            for j in range(n_chans):
                                mean_corr1[key_idx,i,j]=np.nanmean(corr_val_x1[key_idx,:,i,j])
                                if x2 is not None:
                                    mean_corr2[key_idx,i,j]=np.nanmean(corr_val_x2[key_idx,:,i,j])
                        key_idx+=1
                    if x2 is None:
                        vmax=np.nanmax(mean_corr1)
                    else:
                        vmax=np.nanmax([np.nanmax(mean_corr1),np.nanmax(mean_corr2)])
                    #plot
                    if x2 is None:
                        if len(event_dict.keys())==1:
                            fig, axes = plt.subplots(nrows=1, ncols=1)
                            df=pd.DataFrame(mean_corr1[0,:,:],index=x1.ch_names,columns=x1.ch_names)
                            sns.heatmap(df,xticklabels=df.index,yticklabels=df.columns,mask=df.isnull(),cmap='Reds',ax=axes,vmin=0,vmax=vmax)
                            plt.xlabel("Electrodes")
                            plt.ylabel("Electrodes")
                            axes.figure.axes[-1].set_ylabel('Transfer entropy\n(average across epochs)',size=14)
                            plt.show()
                        elif len(event_dict.keys())>2:
                            fig,axes = plt.subplots(nrows=len(event_dict.keys()))
                            dfs=[]
                            rows=list(event_dict.keys())
                            for i in range(len(event_dict.keys())):
                                dfs.append(pd.DataFrame(mean_corr1[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                            pad=5
                            for ax, row in zip(axes[:], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for i in range(len(event_dict.keys())):
                                sns.heatmap(dfs[i],xticklabels=dfs[i].index,yticklabels=dfs[i].columns,mask=dfs[i].isnull(),cmap='Reds',ax=axes[i],vmin=0,vmax=vmax)
                                axes[i].set_ylabel('Electrodes')
                                axes[i].set_xlabel('Electrodes')
                                axes[i].figure.axes[-1].set_ylabel('Transfer entropy\n(average across epochs)')
                            plt.show()
                        elif len(event_dict.keys())==2:
                            fig,axes = plt.subplots(nrows=3)
                            dfs=[]
                            rows=list(event_dict.keys())+['Difference\n'+list(event_dict.keys())[1]+'\n'+list(event_dict.keys())[0]]
                            for i in range(2):
                                dfs.append(pd.DataFrame(mean_corr1[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                            dfs.append(pd.DataFrame(dfs[1]-dfs[0]))
                            pad=5
                            for ax, row in zip(axes[:], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for i in range(len(rows)):
                                sns.heatmap(dfs[i],xticklabels=dfs[i].index,yticklabels=dfs[i].columns,mask=dfs[i].isnull(),cmap='Reds',ax=axes[i],vmin=0,vmax=vmax)
                                axes[i].set_ylabel('Electrodes')
                                axes[i].set_xlabel('Electrodes')
                                if i==len(rows)-1:
                                    axes[i].figure.axes[-1].set_ylabel('Difference of\ntransfer entropy values')
                                else:
                                    axes[i].figure.axes[-1].set_ylabel('Transfer entropy\n(average across epochs)')
                            plt.show()
                    else:
                        if len(event_dict.keys())==1:
                            fig, axes = plt.subplots(nrows=1, ncols=3)
                            df1=pd.DataFrame(mean_corr1[0,:,:],index=x1.ch_names,columns=x1.ch_names)
                            df2=pd.DataFrame(mean_corr2[0,:,:],index=x2.ch_names,columns=x2.ch_names)
                            df_diff=pd.DataFrame(df2-df1)
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            pad=5
                            for ax, col in zip(axes[:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            sns.heatmap(df1,xticklabels=df1.index,yticklabels=df1.columns,mask=df1.isnull(),cmap='Reds',ax=axes[0],vmin=0,vmax=vmax)
                            axes[0].set_ylabel('Electrodes')
                            axes[0].set_xlabel('Electrodes')
                            axes[0].figure.axes[-1].set_ylabel('Transfer entropy\n(average across epochs)')
                            sns.heatmap(df2,xticklabels=df2.index,yticklabels=df2.columns,mask=df2.isnull(),cmap='Reds',ax=axes[1],vmin=0,vmax=vmax)
                            axes[1].set_ylabel('Electrodes')
                            axes[1].set_xlabel('Electrodes')
                            axes[1].figure.axes[-1].set_ylabel('Transfer entropy\n(average across epochs)')
                            sns.heatmap(df_diff,xticklabels=df_diff.index,yticklabels=df_diff.columns,mask=df_diff.isnull(),cmap='Reds',ax=axes[2],vmin=0,vmax=vmax)
                            axes[2].set_ylabel('Electrodes')
                            axes[2].set_xlabel('Electrodes')
                            axes[2].figure.axes[-1].set_ylabel('Difference of\nTransfer entropy values')
                            plt.show()
                        elif len(event_dict.keys())>2:
                            fig,axes = plt.subplots(nrows=len(event_dict.keys()),ncols=3)
                            dfs1=[]
                            dfs2=[]
                            dfs_diff=[]
                            rows=list(event_dict.keys())
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            for i in range(len(event_dict.keys())):
                                dfs1.append(pd.DataFrame(mean_corr1[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                                dfs2.append(pd.DataFrame(mean_corr2[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                                dfs_diff.append(pd.DataFrame(dfs2[i]-dfs1[i]))
                            pad=5
                            for ax, row in zip(axes[:,0], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',size=17,ha='right',va='center')
                            for ax, col in zip(axes[0,:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            for i in range(len(event_dict.keys())):
                                sns.heatmap(dfs1[i],xticklabels=dfs1[i].index,yticklabels=dfs1[i].columns,mask=dfs1[i].isnull(),cmap='Reds',ax=axes[i,0],vmin=0,vmax=vmax)
                                axes[i,0].set_ylabel('Electrodes')
                                axes[i,0].set_xlabel('Electrodes')
                                axes[i,0].figure.axes[-1].set_ylabel('Transfer entropy\n(average across epochs)')
                                sns.heatmap(df2,xticklabels=dfs2[i].index,yticklabels=dfs2[i].columns,mask=dfs2[i].isnull(),cmap='Reds',ax=axes[i,1],vmin=0,vmax=vmax)
                                axes[i,1].set_ylabel('Electrodes')
                                axes[i,1].set_xlabel('Electrodes')
                                axes[i,1].figure.axes[-1].set_ylabel('Transfer entropy\n(average across epochs)')
                                sns.heatmap(dfs_diff[i],xticklabels=dfs_diff[i].index,yticklabels=dfs_diff[i].columns,mask=dfs_diff[i].isnull(),cmap='Reds',ax=axes[i,2],vmin=0,vmax=vmax)
                                axes[i,2].set_ylabel('Electrodes')
                                axes[i,2].set_xlabel('Electrodes')
                                axes[i,2].figure.axes[-1].set_ylabel('Difference of\nTransfer entropy values')
                            plt.show()
                        elif len(event_dict.keys())==2:
                            fig,axes = plt.subplots(nrows=3,ncols=3)
                            dfs1=[]
                            dfs2=[]
                            dfs_diff=[]
                            rows=list(event_dict.keys())+['Difference\n'+list(event_dict.keys())[1]+'\n'+list(event_dict.keys())[0]]
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            for i in range(2):
                                dfs1.append(pd.DataFrame(mean_corr1[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                                dfs2.append(pd.DataFrame(mean_corr2[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                                dfs_diff.append(pd.DataFrame(dfs2[i]-dfs1[i]))
                            dfs1.append(pd.DataFrame(dfs1[1]-dfs1[0]))
                            dfs2.append(pd.DataFrame(dfs2[1]-dfs2[0]))
                            pad=5
                            for ax, row in zip(axes[:,0], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for ax, col in zip(axes[0,:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            for i in range(3):
                                sns.heatmap(dfs1[i],xticklabels=dfs1[i].index,yticklabels=dfs1[i].columns,mask=dfs1[i].isnull(),cmap='Reds',ax=axes[i,0],vmin=0,vmax=vmax)
                                axes[i,0].set_ylabel('Electrodes')
                                axes[i,0].set_xlabel('Electrodes')
                                if i<2:
                                    axes[i,0].figure.axes[-1].set_ylabel('Transfer entropy\n(average across epochs)')
                                else:
                                    axes[i,0].figure.axes[-1].set_ylabel('Difference of\nTransfer entropy values')
                                sns.heatmap(dfs2[i],xticklabels=dfs2[i].index,yticklabels=dfs2[i].columns,mask=dfs2[i].isnull(),cmap='Reds',ax=axes[i,1],vmin=0,vmax=vmax)
                                axes[i,1].set_ylabel('Electrodes')
                                axes[i,1].set_xlabel('Electrodes')
                                if i<2:
                                    axes[i,1].figure.axes[-1].set_ylabel('Transfer entropy\n(average across epochs)')
                                else:
                                    axes[i,1].figure.axes[-1].set_ylabel('Difference of\nTransfer entropy values')
                                if i<2:
                                    sns.heatmap(dfs_diff[i],xticklabels=dfs_diff[i].index,yticklabels=dfs_diff[i].columns,mask=dfs_diff[i].isnull(),cmap='Reds',ax=axes[i,2],vmin=0,vmax=vmax)
                                    axes[i,2].set_ylabel('Electrodes')
                                    axes[i,2].set_xlabel('Electrodes')
                                    axes[i,2].figure.axes[-1].set_ylabel('Difference of\nTransfer entropy values')
                            fig.delaxes(axes[2,2])
                            plt.show()
                Button(win,text="Save results",command=save_corr).grid(row=13,column=0,padx=10,sticky=W)
                Button(win,text="Make plot",command=plot_corr).grid(row=14,column=0,padx=10,sticky=W)
                Button(win,text="Close",command=win.destroy).grid(row=15,column=0,padx=10,pady=10,columnspan=5)
            else:
                showinfo(title="Error",message="Delay must be a positive number")
        def find_optimal_tau():
            event_names=list(event_dict.keys())
            optim_tau=np.empty((1,len(event_names),len(x1.events),len(x1.ch_names)))
            if x2 is not None:
                optim_tau=np.empty((2,len(event_names),max((len(x1.events),len(x2.events))),len(x1.ch_names)))
            optim_tau.fill(np.nan)
            count=0
            pmax2=0
            for k in range(len(event_names)):
                for sel_chan in range(len(x1.ch_names)):
                    for ev in range(len(x1[event_names[k]])):
                        pmax2+=1
                    if x2 is not None:
                        for ev in range(len(x2[event_names[k]])):
                            pmax2+=1
            for k in range(len(event_names)):
                vals1=x1[event_names[k]].get_data()
                for sel_chan in range(len(x1.ch_names)):
                    for ev in range(vals1.shape[0]):
                        count+=1
                        win.update_idletasks()
                        win.update()
                        pbar2['value'] += 100/pmax2
                        pbtxt2['text']=f"{count:d}/{pmax2:d}"
                        optim_tau[0,k,ev,sel_chan]=int(complexity_delay(vals1[ev,sel_chan,:]))
                    if x2 is not None:
                        vals2=x2[event_names[k]].get_data()
                        for ev in range(vals2.shape[0]):
                            count+=1
                            win.update_idletasks()
                            win.update()
                            pbar2['value'] += 100/pmax2
                            pbtxt2['text']=f"{count:d}/{pmax2:d}"
                            optim_tau[1,k,ev,sel_chan]=int(complexity_delay(vals2[ev,sel_chan,:]))
            cond_names=[cond1_name.get()]
            if x2 is not None:
                cond_names.append(cond2_name.get())
            titles=[]
            rows_tau=[]
            for cond in range(len(cond_names)):
                for ev in range(len(event_names)):
                    titles.append(cond_names[cond]+'\n'+event_names[ev])
                    to_append=[optim_tau[cond,ev,ev_count,sel_chan] for ev_count in range(optim_tau.shape[2]) for sel_chan in range(optim_tau.shape[3])]
                    rows_tau.append(to_append)
            tau_df=pd.DataFrame(data=rows_tau,index=titles).T
            plt.figure()
            ax = sns.boxplot(data=tau_df,orient="v")
            ax.set_xlabel("Condition/event")
            ax.set_ylabel("Optimal tau (Takens' reconstruction delay)")
            plt.show()
        Label(win,text="Calculate Transfer Entropy").grid(row=0,column=0,columnspan=5,padx=10,pady=10)
        Label(win,text="Transmission delay between brain regions (ms):").grid(row=1,column=0,padx=10,sticky=W)
        Entry(win,textvariable=delay,width=4).grid(row=1,column=1,sticky=W)
        Label(win,text="Number of symbols (partition divisions):").grid(row=2,column=0,sticky=W,padx=10)
        Entry(win,textvariable=ns,width=3).grid(row=2,column=1,sticky=W)
        Label(win,text="Partition divisions:").grid(row=3,column=0,sticky=W,padx=10)
        Radiobutton(win,text="equal-sized divisions",variable=div_type,value=1).grid(row=3,column=1,sticky=W)
        Radiobutton(win,text="divisions with same number of points",variable=div_type,value=2).grid(row=4,column=1,columnspan=4,sticky=W)
        Radiobutton(win,text="other (separate with commas):",variable=div_type,value=3).grid(row=5,column=1,sticky=W)
        Label(win,text=" X divisions:").grid(row=5,column=2,sticky=W)
        Entry(win,textvariable=xdiv_vals,width=20).grid(row=5,column=3,sticky=W)
        Label(win,text=" Y divisions:").grid(row=6,column=2,sticky=W)
        Entry(win,textvariable=ydiv_vals,width=20).grid(row=6,column=3,sticky=W)
        Label(win,text="Symbolic length:").grid(row=7,column=0,sticky=W,padx=10)
        Label(win,text="Past of X:").grid(row=7,column=1,sticky=W)
        Entry(win,textvariable=lxp,width=3).grid(row=7,column=2,sticky=W)
        Label(win,text="Past of Y:").grid(row=8,column=1,sticky=W)
        Entry(win,textvariable=lyp,width=3).grid(row=8,column=2,sticky=W)
        Label(win,text="Future of Y:").grid(row=9,column=1,sticky=W)
        Entry(win,textvariable=lyf,width=3).grid(row=9,column=2,sticky=W)
        Label(win,text="Tau:").grid(row=10,column=0,sticky=W,padx=10)
        Entry(win,textvariable=tau,width=3).grid(row=10,column=1,sticky=W)
        btn_tau=Button(win,text="Find best tau",command=find_optimal_tau)
        btn_tau.grid(row=10,column=2,sticky=W)
        pbar2=Progressbar(win,orient=HORIZONTAL,length=200,mode='determinate')
        pbar2['value']=0.0
        pbtxt2=Label(win,text="--")
        pbar2.grid(row=10,column=3,sticky=W)
        pbtxt2.grid(row=10,column=4,sticky=W)
        Label(win,text="Units:").grid(row=11,column=0,sticky=W,padx=10)
        OptionMenu(win,unit,*optionlist).grid(row=11,column=1,sticky=W) 
        pbar=Progressbar(win,orient=HORIZONTAL,length=500,mode='determinate')
        pbar['value']=0.0
        pbtxt=Label(win,text="--")
        btn=Button(win, text='Start calculation', command=step)
        btn.grid(row=12,column=0,sticky=W,padx=10)
        pbar.grid(row=12,column=1,sticky=W,columnspan=3)
        pbtxt.grid(row=12,column=4,sticky=W)

#mutual information and comparison between conditions
def mi():
    win=Toplevel(main)
    global x1
    global x2
    delay=StringVar()
    delay.set("10")
    ns=StringVar()
    ns.set('5')
    div_type=IntVar()
    div_type.set(3)
    xdiv_vals=StringVar()    
    ydiv_vals=StringVar()    
    lxp=StringVar()
    lxp.set('1')
    lxp=StringVar()
    lxp.set('1')
    lyp=StringVar()
    lyp.set('1')
    tau=StringVar()
    tau.set('1')
    unit=StringVar()
    optionlist=['bits','nat','ban']
    error=make_x()
    if error==1:
        Label(win,text="ERROR\nIt is required at least\n1 preprocessed data",justify=CENTER).grid(row=0,column=0,padx=10,pady=10)        
        Button(win,text="OK",command=win.destroy).grid(row=1,column=0,padx=10)
    else:
        xdiv_vals.set(f'{np.quantile(eeg1.get_data(),0.2)*10**6:.2f}, {np.quantile(eeg1.get_data(),0.4)*10**6:.2f}, {np.quantile(eeg1.get_data(),0.6)*10**6:.2f}, {np.quantile(eeg1.get_data(),0.8)*10**6:.2f}')
        ydiv_vals.set(f'{np.quantile(eeg1.get_data(),0.2)*10**6:.2f}, {np.quantile(eeg1.get_data(),0.4)*10**6:.2f}, {np.quantile(eeg1.get_data(),0.6)*10**6:.2f}, {np.quantile(eeg1.get_data(),0.8)*10**6:.2f}')
        n_chans=len(eeg1.ch_names)
        def step():
            if int(delay.get())>=0:
                delayval=int(float(delay.get())*eeg1.info['sfreq']/1000)
                max_epo_x1=max([len(x1[key]) for key in event_dict.keys()])
                corr_val_x1=np.empty((len(event_dict.keys()),max_epo_x1,n_chans,n_chans))
                corr_val_x1.fill(np.nan)
                if x2 is not None:
                    max_epo_x2=max([len(x2[key]) for key in event_dict.keys()])
                    corr_val_x2=np.empty((len(event_dict.keys()),max_epo_x2,n_chans,n_chans))
                    corr_val_x2.fill(np.nan)
                if div_type.get()==1:
                    symb_type='equal-divs'
                    x_divs,y_divs=None,None
                elif div_type.get()==2:
                    symb_type='equal-points'
                    x_divs,y_divs=None,None
                elif div_type.get()==3:
                    x_divs=[float(i)*10**-6 for i in xdiv_vals.get().split(sep=',')]
                    y_divs=[float(i)*10**-6 for i in ydiv_vals.get().split(sep=',')]
                    symb_type=None
                total_steps=0
                for i in range(n_chans):
                    for j in range(n_chans):
                        for key in event_dict.keys():
                            for epoch1len in range(len(x1[key])):
                                total_steps+=1
                            if x2 is not None:
                                for epoch2len in range(len(x2[key])):
                                    total_steps+=1            
                pbar['value']=0.0
                k=0                
                for i in range(n_chans):
                    for j in range(n_chans):
                        key_idx=0
                        for key in event_dict.keys():
                            n_epochs=len(x1[key])
                            for epoch in range(n_epochs):
                                win.update_idletasks()
                                win.update()
                                k+=1
                                pbar['value'] += 100/total_steps
                                pbtxt['text']=f"{k:d}/{total_steps:d}"
                                if delayval>0:
                                    corr_val_x1[key_idx,epoch,i,j]=mutual_info(x1[key][epoch]._data[0,i,:-delayval],x1[key][epoch]._data[0,j,delayval:],
                                                                                    symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                    symbolic_length=(int(lxp.get()),int(lyp.get())),
                                                                                    x_divs=x_divs,y_divs=y_divs,units=unit.get())
                                elif delayval==0:
                                    corr_val_x1[key_idx,epoch,i,j]=mutual_info(x1[key][epoch]._data[0,i,:],x1[key][epoch]._data[0,j,:],
                                                                                    symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                    symbolic_length=(int(lxp.get()),int(lyp.get())),
                                                                                    x_divs=x_divs,y_divs=y_divs,units=unit.get())
                                else:
                                    showinfo(title="Error",message="Delay must be positive")
                            if x2 is not None:
                                n_epochs2=len(x2[key])
                                for epoch2 in range(n_epochs2):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar['value'] += 100/total_steps
                                    pbtxt['text']=f"{k:d}/{total_steps:d}"
                                    if delayval>0:
                                        corr_val_x2[key_idx,epoch2,i,j]=mutual_info(x2[key][epoch2]._data[0,i,:-delayval],x2[key][epoch2]._data[0,j,delayval:],
                                                                                          symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                          symbolic_length=(int(lxp.get()),int(lyp.get())),
                                                                                          x_divs=x_divs,y_divs=y_divs,units=unit.get())
                                    elif delayval==0:
                                        corr_val_x2[key_idx,epoch2,i,j]=mutual_info(x2[key][epoch2]._data[0,i,:],x2[key][epoch2]._data[0,j,:],
                                                                                          symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                          symbolic_length=(int(lxp.get()),int(lyp.get())),
                                                                                          x_divs=x_divs,y_divs=y_divs,units=unit.get())
                                    else:
                                        showinfo(title="Error",message="Delay must be positive")
                            key_idx+=1
                def save_corr():
                    showinfo(title="Info",message="First you'll save the average\nacross epochs (CSV table)")
                    key_idx=0
                    for key in event_dict.keys():
                        #calculate mean correlations
                        mean_corr1=np.empty((n_chans,n_chans))
                        mean_corr1.fill(np.nan)
                        if x2 is not None:
                            mean_corr2=np.empty((n_chans,n_chans))
                            mean_corr2.fill(np.nan)
                        for i in range(n_chans):
                            for j in range(n_chans):
                                mean_corr1[i,j]=np.nanmean(corr_val_x1[key_idx,:,i,j])
                                if x2 is not None:
                                    mean_corr2[i,j]=np.nanmean(corr_val_x2[key_idx,:,i,j])
                        #save
                        fname1 = fd.asksaveasfilename(title="MI condition "+cond1_name.get()+" event "+key,defaultextension=".csv",filetypes=(("Comma Separated Values", "*.csv"),("All Files", "*.*")))
                        df=pd.DataFrame(mean_corr1,index=x1.ch_names,columns=x1.ch_names)
                        df.to_csv(fname1)
                        if x2 is not None:
                            fname2 = fd.asksaveasfilename(title="MI condition "+cond2_name.get()+" event "+key,defaultextension=".csv",filetypes=(("Comma Separated Values", "*.csv"),("All Files", "*.*")))
                            df2=pd.DataFrame(mean_corr2,index=x2.ch_names,columns=x2.ch_names)
                            df2.to_csv(fname2)
                        key_idx+=1
                    #save np.array
                    showinfo(title="Info",message="Now you'll save the full results\n(Numpy array of size (n_events,n_epochs,n_chans,n_chans))")
                    fname1 = fd.asksaveasfilename(title="MI condition "+cond1_name.get(),defaultextension=".npy",filetypes=(("Numpy array", "*.npy"),("All Files", "*.*")))
                    np.save(fname1,corr_val_x1)
                    if x2 is not None:
                        fname2 = fd.asksaveasfilename(title="MI condition "+cond2_name.get(),defaultextension=".npy",filetypes=(("Numpy array", "*.npy"),("All Files", "*.*")))
                        np.save(fname2,corr_val_x2)
                def plot_corr():
                    #calculate mean correlations
                    key_idx=0
                    mean_corr1=np.empty((len(event_dict.keys()),n_chans,n_chans))
                    mean_corr1.fill(np.nan)
                    if x2 is not None:
                        mean_corr2=np.empty((len(event_dict.keys()),n_chans,n_chans))
                        mean_corr2.fill(np.nan)
                    for key in event_dict.keys():
                        for i in range(n_chans):
                            for j in range(n_chans):
                                mean_corr1[key_idx,i,j]=np.nanmean(corr_val_x1[key_idx,:,i,j])
                                if x2 is not None:
                                    mean_corr2[key_idx,i,j]=np.nanmean(corr_val_x2[key_idx,:,i,j])
                        key_idx+=1
                    if x2 is None:
                        vmax=np.nanmax(mean_corr1)
                    else:
                        vmax=np.nanmax([np.nanmax(mean_corr1),np.nanmax(mean_corr2)])
                    #plot
                    if x2 is None:
                        if len(event_dict.keys())==1:
                            fig, axes = plt.subplots(nrows=1, ncols=1)
                            df=pd.DataFrame(mean_corr1[0,:,:],index=x1.ch_names,columns=x1.ch_names)
                            sns.heatmap(df,xticklabels=df.index,yticklabels=df.columns,mask=df.isnull(),cmap='Reds',ax=axes,vmin=0,vmax=vmax)
                            plt.xlabel("Electrodes")
                            plt.ylabel("Electrodes")
                            axes.figure.axes[-1].set_ylabel('Mutual information\n(average across epochs)',size=14)
                            plt.show()
                        elif len(event_dict.keys())>2:
                            fig,axes = plt.subplots(nrows=len(event_dict.keys()))
                            dfs=[]
                            rows=list(event_dict.keys())
                            for i in range(len(event_dict.keys())):
                                dfs.append(pd.DataFrame(mean_corr1[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                            pad=5
                            for ax, row in zip(axes[:], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for i in range(len(event_dict.keys())):
                                sns.heatmap(dfs[i],xticklabels=dfs[i].index,yticklabels=dfs[i].columns,mask=dfs[i].isnull(),cmap='Reds',ax=axes[i],vmin=0,vmax=vmax)
                                axes[i].set_ylabel('Electrodes')
                                axes[i].set_xlabel('Electrodes')
                                axes[i].figure.axes[-1].set_ylabel('Mutual information\n(average across epochs)')
                            plt.show()
                        elif len(event_dict.keys())==2:
                            fig,axes = plt.subplots(nrows=3)
                            dfs=[]
                            rows=list(event_dict.keys())+['Difference\n'+list(event_dict.keys())[1]+'\n'+list(event_dict.keys())[0]]
                            for i in range(2):
                                dfs.append(pd.DataFrame(mean_corr1[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                            dfs.append(pd.DataFrame(dfs[1]-dfs[0]))
                            pad=5
                            for ax, row in zip(axes[:], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for i in range(len(rows)):
                                sns.heatmap(dfs[i],xticklabels=dfs[i].index,yticklabels=dfs[i].columns,mask=dfs[i].isnull(),cmap='Reds',ax=axes[i],vmin=0,vmax=vmax)
                                axes[i].set_ylabel('Electrodes')
                                axes[i].set_xlabel('Electrodes')
                                if i==len(rows)-1:
                                    axes[i].figure.axes[-1].set_ylabel('Difference of\nmutual information values')
                                else:
                                    axes[i].figure.axes[-1].set_ylabel('Mutual information\n(average across epochs)')
                            plt.show()
                    else:
                        if len(event_dict.keys())==1:
                            fig, axes = plt.subplots(nrows=1, ncols=3)
                            df1=pd.DataFrame(mean_corr1[0,:,:],index=x1.ch_names,columns=x1.ch_names)
                            df2=pd.DataFrame(mean_corr2[0,:,:],index=x2.ch_names,columns=x2.ch_names)
                            df_diff=pd.DataFrame(df2-df1)
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            pad=5
                            for ax, col in zip(axes[:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            sns.heatmap(df1,xticklabels=df1.index,yticklabels=df1.columns,mask=df1.isnull(),cmap='Reds',ax=axes[0],vmin=0,vmax=vmax)
                            axes[0].set_ylabel('Electrodes')
                            axes[0].set_xlabel('Electrodes')
                            axes[0].figure.axes[-1].set_ylabel('Mutual information\n(average across epochs)')
                            sns.heatmap(df2,xticklabels=df2.index,yticklabels=df2.columns,mask=df2.isnull(),cmap='Reds',ax=axes[1],vmin=0,vmax=vmax)
                            axes[1].set_ylabel('Electrodes')
                            axes[1].set_xlabel('Electrodes')
                            axes[1].figure.axes[-1].set_ylabel('Mutual information\n(average across epochs)')
                            sns.heatmap(df_diff,xticklabels=df_diff.index,yticklabels=df_diff.columns,mask=df_diff.isnull(),cmap='Reds',ax=axes[2],vmin=0,vmax=vmax)
                            axes[2].set_ylabel('Electrodes')
                            axes[2].set_xlabel('Electrodes')
                            axes[2].figure.axes[-1].set_ylabel('Difference of\nmutual information values')
                            plt.show()
                        elif len(event_dict.keys())>2:
                            fig,axes = plt.subplots(nrows=len(event_dict.keys()),ncols=3)
                            dfs1=[]
                            dfs2=[]
                            dfs_diff=[]
                            rows=list(event_dict.keys())
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            for i in range(len(event_dict.keys())):
                                dfs1.append(pd.DataFrame(mean_corr1[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                                dfs2.append(pd.DataFrame(mean_corr2[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                                dfs_diff.append(pd.DataFrame(dfs2[i]-dfs1[i]))
                            pad=5
                            for ax, row in zip(axes[:,0], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',size=17,ha='right',va='center')
                            for ax, col in zip(axes[0,:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            for i in range(len(event_dict.keys())):
                                sns.heatmap(dfs1[i],xticklabels=dfs1[i].index,yticklabels=dfs1[i].columns,mask=dfs1[i].isnull(),cmap='Reds',ax=axes[i,0],vmin=0,vmax=vmax)
                                axes[i,0].set_ylabel('Electrodes')
                                axes[i,0].set_xlabel('Electrodes')
                                axes[i,0].figure.axes[-1].set_ylabel('Mutual information\n(average across epochs)')
                                sns.heatmap(dfs2[i],xticklabels=dfs2[i].index,yticklabels=dfs2[i].columns,mask=dfs2[i].isnull(),cmap='Reds',ax=axes[i,1],vmin=0,vmax=vmax)
                                axes[i,1].set_ylabel('Electrodes')
                                axes[i,1].set_xlabel('Electrodes')
                                axes[i,1].figure.axes[-1].set_ylabel('Mutual information\n(average across epochs)')
                                sns.heatmap(dfs_diff[i],xticklabels=dfs_diff[i].index,yticklabels=dfs_diff[i].columns,mask=dfs_diff[i].isnull(),cmap='Reds',ax=axes[i,2],vmin=0,vmax=vmax)
                                axes[i,2].set_ylabel('Electrodes')
                                axes[i,2].set_xlabel('Electrodes')
                                axes[i,2].figure.axes[-1].set_ylabel('Difference of\nmutual information values')
                            plt.show()
                        elif len(event_dict.keys())==2:
                            fig,axes = plt.subplots(nrows=3,ncols=3)
                            dfs1=[]
                            dfs2=[]
                            dfs_diff=[]
                            rows=list(event_dict.keys())+['Difference\n'+list(event_dict.keys())[1]+'\n'+list(event_dict.keys())[0]]
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            for i in range(2):
                                dfs1.append(pd.DataFrame(mean_corr1[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                                dfs2.append(pd.DataFrame(mean_corr2[i,:,:],index=x1.ch_names,columns=x1.ch_names))
                                dfs_diff.append(pd.DataFrame(dfs2[i]-dfs1[i]))
                            dfs1.append(pd.DataFrame(dfs1[1]-dfs1[0]))
                            dfs2.append(pd.DataFrame(dfs2[1]-dfs2[0]))
                            pad=5
                            for ax, row in zip(axes[:,0], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for ax, col in zip(axes[0,:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            for i in range(3):
                                sns.heatmap(dfs1[i],xticklabels=dfs1[i].index,yticklabels=dfs1[i].columns,mask=dfs1[i].isnull(),cmap='Reds',ax=axes[i,0],vmin=0,vmax=vmax)
                                axes[i,0].set_ylabel('Electrodes')
                                axes[i,0].set_xlabel('Electrodes')
                                if i<2:
                                    axes[i,0].figure.axes[-1].set_ylabel('Mutual information\n(average across epochs)')
                                else:
                                    axes[i,0].figure.axes[-1].set_ylabel('Difference of\nmutual information values')
                                sns.heatmap(dfs2[i],xticklabels=dfs2[i].index,yticklabels=dfs2[i].columns,mask=dfs2[i].isnull(),cmap='Reds',ax=axes[i,1],vmin=0,vmax=vmax)
                                axes[i,1].set_ylabel('Electrodes')
                                axes[i,1].set_xlabel('Electrodes')
                                if i<2:
                                    axes[i,1].figure.axes[-1].set_ylabel('Mutual information\n(average across epochs)')
                                else:
                                    axes[i,1].figure.axes[-1].set_ylabel('Difference of\nmutual information values')
                                if i<2:
                                    sns.heatmap(dfs_diff[i],xticklabels=dfs_diff[i].index,yticklabels=dfs_diff[i].columns,mask=dfs_diff[i].isnull(),cmap='Reds',ax=axes[i,2],vmin=0,vmax=vmax)
                                    axes[i,2].set_ylabel('Electrodes')
                                    axes[i,2].set_xlabel('Electrodes')
                                    axes[i,2].figure.axes[-1].set_ylabel('Difference of\nmutual information values')
                            fig.delaxes(axes[2,2])
                            plt.show()
                Button(win,text="Save results",command=save_corr).grid(row=12,column=0,padx=10,sticky=W)
                Button(win,text="Make plot",command=plot_corr).grid(row=13,column=0,padx=10,sticky=W)
                Button(win,text="Close",command=win.destroy).grid(row=14,column=0,padx=10,pady=10,columnspan=5)
            else:
                showinfo(title="Error",message="Delay must be a positive number")
        def find_optimal_tau():
            event_names=list(event_dict.keys())
            optim_tau=np.empty((1,len(event_names),len(x1.events),len(x1.ch_names)))
            if x2 is not None:
                optim_tau=np.empty((2,len(event_names),max((len(x1.events),len(x2.events))),len(x1.ch_names)))
            optim_tau.fill(np.nan)
            count=0
            pmax2=0
            for k in range(len(event_names)):
                for sel_chan in range(len(x1.ch_names)):
                    for ev in range(len(x1[event_names[k]])):
                        pmax2+=1
                    if x2 is not None:
                        for ev in range(len(x2[event_names[k]])):
                            pmax2+=1
            for k in range(len(event_names)):
                vals1=x1[event_names[k]].get_data()
                for sel_chan in range(len(x1.ch_names)):
                    for ev in range(vals1.shape[0]):
                        count+=1
                        win.update_idletasks()
                        win.update()
                        pbar2['value'] += 100/pmax2
                        pbtxt2['text']=f"{count:d}/{pmax2:d}"
                        optim_tau[0,k,ev,sel_chan]=int(complexity_delay(vals1[ev,sel_chan,:]))
                    if x2 is not None:
                        vals2=x2[event_names[k]].get_data()
                        for ev in range(vals2.shape[0]):
                            count+=1
                            win.update_idletasks()
                            win.update()
                            pbar2['value'] += 100/pmax2
                            pbtxt2['text']=f"{count:d}/{pmax2:d}"
                            optim_tau[1,k,ev,sel_chan]=int(complexity_delay(vals2[ev,sel_chan,:]))
            cond_names=[cond1_name.get()]
            if x2 is not None:
                cond_names.append(cond2_name.get())
            titles=[]
            rows_tau=[]
            for cond in range(len(cond_names)):
                for ev in range(len(event_names)):
                    titles.append(cond_names[cond]+'\n'+event_names[ev])
                    to_append=[optim_tau[cond,ev,ev_count,sel_chan] for ev_count in range(optim_tau.shape[2]) for sel_chan in range(optim_tau.shape[3])]
                    rows_tau.append(to_append)
            tau_df=pd.DataFrame(data=rows_tau,index=titles).T
            plt.figure()
            ax = sns.boxplot(data=tau_df,orient="v")
            ax.set_xlabel("Condition/event")
            ax.set_ylabel("Optimal tau (Takens' reconstruction delay)")
            plt.show()
        Label(win,text="Calculate Mutual Information").grid(row=0,column=0,columnspan=5,padx=10,pady=10)
        Label(win,text="Transmission delay between brain regions (ms):").grid(row=1,column=0,padx=10,sticky=W)
        Entry(win,textvariable=delay,width=4).grid(row=1,column=1,sticky=W)
        Label(win,text="Number of symbols (partition divisions):").grid(row=2,column=0,sticky=W,padx=10)
        Entry(win,textvariable=ns,width=3).grid(row=2,column=1,sticky=W)
        Label(win,text="Partition divisions:").grid(row=3,column=0,sticky=W,padx=10)
        Radiobutton(win,text="equal-sized divisions",variable=div_type,value=1).grid(row=3,column=1,sticky=W)
        Radiobutton(win,text="divisions with same number of points",variable=div_type,value=2).grid(row=4,column=1,columnspan=4,sticky=W)
        Radiobutton(win,text="other (separate with commas):",variable=div_type,value=3).grid(row=5,column=1,sticky=W)
        Label(win,text=" X divisions:").grid(row=5,column=2,sticky=W)
        Entry(win,textvariable=xdiv_vals,width=20).grid(row=5,column=3,sticky=W)
        Label(win,text=" Y divisions:").grid(row=6,column=2,sticky=W)
        Entry(win,textvariable=ydiv_vals,width=20).grid(row=6,column=3,sticky=W)
        Label(win,text="Symbolic length:").grid(row=7,column=0,sticky=W,padx=10)
        Label(win,text="X:").grid(row=7,column=1,sticky=W)
        Entry(win,textvariable=lxp,width=3).grid(row=7,column=2,sticky=W)
        Label(win,text="Y:").grid(row=8,column=1,sticky=W)
        Entry(win,textvariable=lyp,width=3).grid(row=8,column=2,sticky=W)
        Label(win,text="Tau:").grid(row=9,column=0,sticky=W,padx=10)
        Entry(win,textvariable=tau,width=3).grid(row=9,column=1,sticky=W)
        btn_tau=Button(win,text="Find best tau",command=find_optimal_tau)
        btn_tau.grid(row=9,column=2,sticky=W)
        pbar2=Progressbar(win,orient=HORIZONTAL,length=200,mode='determinate')
        pbar2['value']=0.0
        pbtxt2=Label(win,text="--")
        pbar2.grid(row=9,column=3,sticky=W)
        pbtxt2.grid(row=9,column=4,sticky=W)        
        Label(win,text="Units:").grid(row=10,column=0,sticky=W,padx=10)
        OptionMenu(win,unit,*optionlist).grid(row=10,column=1,sticky=W) 
        pbar=Progressbar(win,orient=HORIZONTAL,length=500,mode='determinate')
        pbar['value']=0.0
        pbtxt=Label(win,text="--")
        btn=Button(win, text='Start calculation', command=step)
        btn.grid(row=11,column=0,sticky=W,padx=10)
        pbar.grid(row=11,column=1,sticky=W,columnspan=3)
        pbtxt.grid(row=11,column=4,sticky=W)

#spectral coherence and comparison between conditions
def coherence():
    win=Toplevel(main)
    global x1
    global x2
    fmax=StringVar()
    fmax.set("12")
    fmin=StringVar()
    fmin.set("10")
    error=make_x()
    if error==1:
        Label(win,text="ERROR\nIt is required at least\n1 preprocessed data",justify=CENTER).grid(row=0,column=0,padx=10,pady=10)        
        Button(win,text="OK",command=win.destroy).grid(row=1,column=0,padx=10)
    else:
        n_chans=len(eeg1.ch_names)
        Label(win,text="Calculate correlations").grid(row=0,column=0,columnspan=3,padx=10,pady=10)
        Label(win,text="Minimum frequency (Hz):").grid(row=1,column=0,padx=10,sticky=W)
        Entry(win,textvariable=fmin,width=4).grid(row=1,column=1,sticky=W)
        Label(win,text="Maximum frequency (Hz):").grid(row=2,column=0,padx=10,sticky=W)
        Entry(win,textvariable=fmax,width=4).grid(row=2,column=1,sticky=W)
        pbar=Progressbar(win,orient=HORIZONTAL,length=100,mode='determinate')
        pbar['value']=0.0
        pbtxt=Label(win,text="--")
        def step():
            if (float(fmin.get())>0) and (float(fmax.get())>=float(fmin.get())):
                n_cond=1
                if x2 is not None:
                    n_cond=2
                pbar['value']=0.0
                k=0
                key_idx=0
                coh1=[0 for i in event_dict.keys()]
                imcoh1=[0 for i in event_dict.keys()]
                if x2 is not None:
                    coh2=[0 for i in event_dict.keys()]
                    imcoh2=[0 for i in event_dict.keys()]
                for key in event_dict.keys(): 
                    win.update_idletasks()
                    win.update()
                    k+=1
                    pbar['value'] += 100/(2*len(event_dict.keys())*n_cond)
                    pbtxt['text']=f"{k:d}/{2*len(event_dict.keys())*n_cond:d}"
                    a=mne_connectivity.spectral_connectivity_epochs(x1[key],indices=None,method='coh',mode='multitaper',sfreq=x1.info['sfreq'],
                                                                     fmin=float(fmin.get()),fmax=float(fmax.get()),faverage=True,mt_adaptive=False,n_jobs=1,verbose='ERROR')
                    coh1[key_idx]=np.reshape(a.get_data(),(n_chans,n_chans))
                    coh1[key_idx][:,:]=coh1[key_idx][:,:]+coh1[key_idx][:,:].T
                    win.update_idletasks()
                    win.update()
                    k+=1
                    pbar['value'] += 100/(2*len(event_dict.keys())*n_cond)
                    pbtxt['text']=f"{k:d}/{2*len(event_dict.keys())*n_cond:d}"
                    b=mne_connectivity.spectral_connectivity_epochs(x1[key],indices=None,method='imcoh',mode='multitaper',sfreq=x1.info['sfreq'],
                                                                     fmin=float(fmin.get()),fmax=float(fmax.get()),faverage=True,mt_adaptive=False,n_jobs=1,verbose='ERROR')
                    imcoh1[key_idx]=np.reshape(b.get_data(),(n_chans,n_chans))
                    imcoh1[key_idx][:,:]=imcoh1[key_idx][:,:]-imcoh1[key_idx][:,:].T
                    if x2 is not None:
                        win.update_idletasks()
                        win.update()
                        k+=1
                        pbar['value'] += 100/(2*len(event_dict.keys())*n_cond)
                        pbtxt['text']=f"{k:d}/{2*len(event_dict.keys())*n_cond:d}"
                        c=mne_connectivity.spectral_connectivity_epochs(x2[key],indices=None,method='coh',mode='multitaper',sfreq=x2.info['sfreq'],
                                                                     fmin=float(fmin.get()),fmax=float(fmax.get()),faverage=True,mt_adaptive=False,n_jobs=1,verbose='ERROR')
                        coh2[key_idx]=np.reshape(c.get_data(),(n_chans,n_chans))
                        coh2[key_idx][:,:]=coh2[key_idx][:,:]+coh2[key_idx][:,:].T
                        win.update_idletasks()
                        win.update()
                        k+=1
                        pbar['value'] += 100/(2*len(event_dict.keys())*n_cond)
                        pbtxt['text']=f"{k:d}/{2*len(event_dict.keys())*n_cond:d}"
                        d=mne_connectivity.spectral_connectivity_epochs(x2[key],indices=None,method='imcoh',mode='multitaper',sfreq=x2.info['sfreq'],
                                                                     fmin=float(fmin.get()),fmax=float(fmax.get()),faverage=True,mt_adaptive=False,n_jobs=1,verbose='ERROR')
                        imcoh2[key_idx]=np.reshape(d.get_data(),(n_chans,n_chans))
                        imcoh2[key_idx][:,:]=imcoh2[key_idx][:,:]-imcoh2[key_idx][:,:].T
                    key_idx+=1
                coh1=[pd.DataFrame(coh1[key_idx][:,:],columns=x1.ch_names,index=x1.ch_names) for key_idx in range(len(event_dict.keys()))]
                imcoh1=[pd.DataFrame(imcoh1[key_idx][:,:],columns=x1.ch_names,index=x1.ch_names) for key_idx in range(len(event_dict.keys()))]
                if x2 is not None:
                    coh2=[pd.DataFrame(coh2[key_idx][:,:],columns=x2.ch_names,index=x2.ch_names) for key_idx in range(len(event_dict.keys()))]
                    imcoh2=[pd.DataFrame(imcoh2[key_idx][:,:],columns=x2.ch_names,index=x2.ch_names) for key_idx in range(len(event_dict.keys()))]                
                def save_corr():
                    showinfo(title="Info",message="First you'll save the traditional coherence\nof both conditions (if available) and only after\nthat the imaginary coherence (CSV tables)")
                    key_idx=0
                    for key in event_dict.keys():
                        #save coh
                        fname1 = fd.asksaveasfilename(title="Coherence for condition "+cond1_name.get()+" event "+key,defaultextension=".csv",filetypes=(("Comma Separated Values", "*.csv"),("All Files", "*.*")))
                        df=pd.DataFrame(coh1[key_idx],index=x1.ch_names,columns=x1.ch_names)
                        df.to_csv(fname1)
                        if x2 is not None:
                            fname2 = fd.asksaveasfilename(title="Coherence for condition "+cond2_name.get()+" event "+key,defaultextension=".csv",filetypes=(("Comma Separated Values", "*.csv"),("All Files", "*.*")))
                            df2=pd.DataFrame(coh2[key_idx],index=x2.ch_names,columns=x2.ch_names)
                            df2.to_csv(fname2)
                        key_idx+=1
                    showinfo(title="Info",message="Now you'll save the imaginary coherence (CSV table)")
                    key_idx=0
                    for key in event_dict.keys():
                        #save imcoh
                        fname1 = fd.asksaveasfilename(title="Imaginary coherence for condition "+cond1_name.get()+" event "+key,defaultextension=".csv",filetypes=(("Comma Separated Values", "*.csv"),("All Files", "*.*")))
                        df=pd.DataFrame(imcoh1[key_idx],index=x1.ch_names,columns=x1.ch_names)
                        df.to_csv(fname1)
                        if x2 is not None:
                            fname2 = fd.asksaveasfilename(title="Imaginary coherence for condition "+cond2_name.get()+" event "+key,defaultextension=".csv",filetypes=(("Comma Separated Values", "*.csv"),("All Files", "*.*")))
                            df2=pd.DataFrame(imcoh2[key_idx],index=x2.ch_names,columns=x2.ch_names)
                            df2.to_csv(fname2)
                        key_idx+=1
                def plot_coh():
                    if x2 is None:
                        if len(event_dict.keys())==1:
                            fig, axes = plt.subplots(nrows=1, ncols=1)
                            sns.heatmap(coh1[0],xticklabels=coh1[0].index,yticklabels=coh1[0].columns,mask=coh1[0].isnull(),cmap='Reds',ax=axes,vmin=0,vmax=1)
                            plt.xlabel("Electrodes")
                            plt.ylabel("Electrodes")
                            axes.figure.axes[-1].set_ylabel('Coherence',size=14)
                            plt.show()
                        elif len(event_dict.keys())>2:
                            fig,axes = plt.subplots(nrows=len(event_dict.keys()))
                            rows=list(event_dict.keys())
                            pad=5
                            for ax, row in zip(axes[:], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for i in range(len(event_dict.keys())):
                                sns.heatmap(coh1[i],xticklabels=coh1[i].index,yticklabels=coh1[i].columns,mask=coh1[i].isnull(),cmap='Reds',ax=axes[i],vmin=0,vmax=1)
                                axes[i].set_ylabel('Electrodes')
                                axes[i].set_xlabel('Electrodes')
                                axes[i].figure.axes[-1].set_ylabel('Coherence')
                            plt.show()
                        elif len(event_dict.keys())==2:
                            fig,axes = plt.subplots(nrows=3)
                            rows=list(event_dict.keys())+['Difference\n'+list(event_dict.keys())[1]+'\n'+list(event_dict.keys())[0]]
                            pad=5
                            for ax, row in zip(axes[:], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for i in range(len(rows)):
                                if i==len(rows)-1:
                                    sns.heatmap(coh1[1]-coh1[0],xticklabels=coh1[0].index,yticklabels=coh1[0].columns,mask=coh1[0].isnull(),cmap='bwr',ax=axes[i],vmin=-1,vmax=1)
                                else:
                                    sns.heatmap(coh1[i],xticklabels=coh1[i].index,yticklabels=coh1[i].columns,mask=coh1[i].isnull(),cmap='Reds',ax=axes[i],vmin=0,vmax=1)
                                axes[i].set_ylabel('Electrodes')
                                axes[i].set_xlabel('Electrodes')
                                if i==len(rows)-1:
                                    axes[i].figure.axes[-1].set_ylabel('Difference of\ncoherence values')
                                else:
                                    axes[i].figure.axes[-1].set_ylabel('Coherence')
                            plt.show()
                    else:
                        if len(event_dict.keys())==1:
                            fig, axes = plt.subplots(nrows=1, ncols=3)
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            pad=5
                            for ax, col in zip(axes[:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            sns.heatmap(coh1[0],xticklabels=coh1[0].index,yticklabels=coh1[0].columns,mask=coh1[0].isnull(),cmap='Reds',ax=axes[0],vmin=0,vmax=1)
                            axes[0].set_ylabel('Electrodes')
                            axes[0].set_xlabel('Electrodes')
                            axes[0].figure.axes[-1].set_ylabel('Coherence')
                            sns.heatmap(coh2[0],xticklabels=coh2[0].index,yticklabels=coh2[0].columns,mask=coh2[0].isnull(),cmap='Reds',ax=axes[1],vmin=0,vmax=1)
                            axes[1].set_ylabel('Electrodes')
                            axes[1].set_xlabel('Electrodes')
                            axes[1].figure.axes[-1].set_ylabel('Coherence')
                            sns.heatmap(coh2[0]-coh1[0],xticklabels=coh1[0].index,yticklabels=coh1[0].columns,mask=coh1[0].isnull(),cmap='bwr',ax=axes[2],vmin=-1,vmax=1)
                            axes[2].set_ylabel('Electrodes')
                            axes[2].set_xlabel('Electrodes')
                            axes[2].figure.axes[-1].set_ylabel('Difference of\ncoherence values')
                            plt.show()
                        elif len(event_dict.keys())>2:
                            fig,axes = plt.subplots(nrows=len(event_dict.keys()),ncols=3)
                            rows=list(event_dict.keys())
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            pad=5
                            for ax, row in zip(axes[:,0], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',size=17,ha='right',va='center')
                            for ax, col in zip(axes[0,:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            for i in range(len(event_dict.keys())):
                                sns.heatmap(coh1[i],xticklabels=coh1[i].index,yticklabels=coh1[i].columns,mask=coh1[i].isnull(),cmap='Reds',ax=axes[i,0],vmin=0,vmax=1)
                                axes[i,0].set_ylabel('Electrodes')
                                axes[i,0].set_xlabel('Electrodes')
                                axes[i,0].figure.axes[-1].set_ylabel('Coherence')
                                sns.heatmap(coh2[i],xticklabels=coh2[i].index,yticklabels=coh2[i].columns,mask=coh2[i].isnull(),cmap='Reds',ax=axes[i,1],vmin=0,vmax=1)
                                axes[i,1].set_ylabel('Electrodes')
                                axes[i,1].set_xlabel('Electrodes')
                                axes[i,1].figure.axes[-1].set_ylabel('Coherence')
                                sns.heatmap(coh2[i]-coh1[i],xticklabels=coh1[i].index,yticklabels=coh1[i].columns,mask=coh1[i].isnull(),cmap='bwr',ax=axes[i,2],vmin=-1,vmax=1)
                                axes[i,2].set_ylabel('Electrodes')
                                axes[i,2].set_xlabel('Electrodes')
                                axes[i,2].figure.axes[-1].set_ylabel('Difference of\ncoherence values')
                            plt.show()
                        elif len(event_dict.keys())==2:
                            fig,axes = plt.subplots(nrows=3,ncols=3)
                            rows=list(event_dict.keys())+['Difference\n'+list(event_dict.keys())[1]+'\n'+list(event_dict.keys())[0]]
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            pad=5
                            for ax, row in zip(axes[:,0], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for ax, col in zip(axes[0,:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            for i in range(3):
                                if i<2:
                                    sns.heatmap(coh1[i],xticklabels=coh1[i].index,yticklabels=coh1[i].columns,mask=coh1[i].isnull(),cmap='Reds',ax=axes[i,0],vmin=0,vmax=1)
                                else:
                                    sns.heatmap(coh1[1]-coh1[0],xticklabels=coh1[0].index,yticklabels=coh1[0].columns,mask=coh1[0].isnull(),cmap='bwr',ax=axes[i,0],vmin=-1,vmax=1)
                                axes[i,0].set_ylabel('Electrodes')
                                axes[i,0].set_xlabel('Electrodes')
                                if i<2:
                                    axes[i,0].figure.axes[-1].set_ylabel('Coherence')
                                else:
                                    axes[i,0].figure.axes[-1].set_ylabel('Difference of\ncoherence values')
                                if i<2:
                                    sns.heatmap(coh2[i],xticklabels=coh2[i].index,yticklabels=coh2[i].columns,mask=coh2[i].isnull(),cmap='Reds',ax=axes[i,1],vmin=0,vmax=1)
                                else:
                                    sns.heatmap(coh2[1]-coh2[0],xticklabels=coh2[0].index,yticklabels=coh2[0].columns,mask=coh2[0].isnull(),cmap='bwr',ax=axes[i,1],vmin=-1,vmax=1)
                                axes[i,1].set_ylabel('Electrodes')
                                axes[i,1].set_xlabel('Electrodes')
                                if i<2:
                                    axes[i,1].figure.axes[-1].set_ylabel('Coherence')
                                else:
                                    axes[i,1].figure.axes[-1].set_ylabel('Difference of\ncoherence values')
                                if i<2:
                                    sns.heatmap(coh2[i]-coh1[i],xticklabels=coh1[i].index,yticklabels=coh1[i].columns,mask=coh1[i].isnull(),cmap='bwr',ax=axes[i,2],vmin=-1,vmax=1)
                                    axes[i,2].set_ylabel('Electrodes')
                                    axes[i,2].set_xlabel('Electrodes')
                                    axes[i,2].figure.axes[-1].set_ylabel('Difference of\ncoherence values')
                            fig.delaxes(axes[2,2])
                            plt.show()
                def plot_imcoh():
                    if x2 is None:
                        if len(event_dict.keys())==1:
                            fig, axes = plt.subplots(nrows=1, ncols=1)
                            sns.heatmap(imcoh1[0],xticklabels=imcoh1[0].index,yticklabels=imcoh1[0].columns,mask=imcoh1[0].isnull(),cmap='bwr',ax=axes,vmin=-1,vmax=1)
                            plt.xlabel("Electrodes")
                            plt.ylabel("Electrodes")
                            axes.figure.axes[-1].set_ylabel('Imaginary coherence',size=14)
                            plt.show()
                        elif len(event_dict.keys())>2:
                            fig,axes = plt.subplots(nrows=len(event_dict.keys()))
                            rows=list(event_dict.keys())
                            pad=5
                            for ax, row in zip(axes[:], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for i in range(len(event_dict.keys())):
                                sns.heatmap(imcoh1[i],xticklabels=imcoh1[i].index,yticklabels=imcoh1[i].columns,mask=imcoh1[i].isnull(),cmap='bwr',ax=axes[i],vmin=-1,vmax=1)
                                axes[i].set_ylabel('Electrodes')
                                axes[i].set_xlabel('Electrodes')
                                axes[i].figure.axes[-1].set_ylabel('Imaginary coherence')
                            plt.show()
                        elif len(event_dict.keys())==2:
                            fig,axes = plt.subplots(nrows=3)
                            rows=list(event_dict.keys())+['Difference\n'+list(event_dict.keys())[1]+'\n'+list(event_dict.keys())[0]]
                            pad=5
                            for ax, row in zip(axes[:], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for i in range(len(rows)):
                                if i==len(rows)-1:
                                    sns.heatmap(imcoh1[1]-imcoh1[0],xticklabels=imcoh1[0].index,yticklabels=imcoh1[0].columns,mask=imcoh1[0].isnull(),cmap='bwr',ax=axes[i],vmin=-1,vmax=1)
                                else:
                                    sns.heatmap(imcoh1[i],xticklabels=imcoh1[i].index,yticklabels=imcoh1[i].columns,mask=imcoh1[i].isnull(),cmap='bwr',ax=axes[i],vmin=-1,vmax=1)
                                axes[i].set_ylabel('Electrodes')
                                axes[i].set_xlabel('Electrodes')
                                if i==len(rows)-1:
                                    axes[i].figure.axes[-1].set_ylabel('Difference of\nimaginary coherence values')
                                else:
                                    axes[i].figure.axes[-1].set_ylabel('Imaginary coherence')
                            plt.show()
                    else:
                        if len(event_dict.keys())==1:
                            fig, axes = plt.subplots(nrows=1, ncols=3)
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            pad=5
                            for ax, col in zip(axes[:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            sns.heatmap(imcoh1[0],xticklabels=imcoh1[0].index,yticklabels=imcoh1[0].columns,mask=imcoh1[0].isnull(),cmap='bwr',ax=axes[0],vmin=-1,vmax=1)
                            axes[0].set_ylabel('Electrodes')
                            axes[0].set_xlabel('Electrodes')
                            axes[0].figure.axes[-1].set_ylabel('Imaginary coherence')
                            sns.heatmap(imcoh2[0],xticklabels=imcoh2[0].index,yticklabels=imcoh2[0].columns,mask=imcoh2[0].isnull(),cmap='bwr',ax=axes[1],vmin=-1,vmax=1)
                            axes[1].set_ylabel('Electrodes')
                            axes[1].set_xlabel('Electrodes')
                            axes[1].figure.axes[-1].set_ylabel('Imaginary coherence')
                            sns.heatmap(imcoh2[0]-imcoh1[0],xticklabels=imcoh1[0].index,yticklabels=imcoh1[0].columns,mask=imcoh1[0].isnull(),cmap='bwr',ax=axes[2],vmin=-1,vmax=1)
                            axes[2].set_ylabel('Electrodes')
                            axes[2].set_xlabel('Electrodes')
                            axes[2].figure.axes[-1].set_ylabel('Difference of\nimaginary coherence values')
                            plt.show()
                        elif len(event_dict.keys())>2:
                            fig,axes = plt.subplots(nrows=len(event_dict.keys()),ncols=3)
                            rows=list(event_dict.keys())
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            pad=5
                            for ax, row in zip(axes[:,0], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',size=17,ha='right',va='center')
                            for ax, col in zip(axes[0,:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            for i in range(len(event_dict.keys())):
                                sns.heatmap(imcoh1[i],xticklabels=imcoh1[i].index,yticklabels=imcoh1[i].columns,mask=imcoh1[i].isnull(),cmap='bwr',ax=axes[i,0],vmin=-1,vmax=1)
                                axes[i,0].set_ylabel('Electrodes')
                                axes[i,0].set_xlabel('Electrodes')
                                axes[i,0].figure.axes[-1].set_ylabel('Imaginary coherence')
                                sns.heatmap(imcoh2[i],xticklabels=imcoh2[i].index,yticklabels=imcoh2[i].columns,mask=imcoh2[i].isnull(),cmap='bwr',ax=axes[i,1],vmin=-1,vmax=1)
                                axes[i,1].set_ylabel('Electrodes')
                                axes[i,1].set_xlabel('Electrodes')
                                axes[i,1].figure.axes[-1].set_ylabel('Imaginary coherence')
                                sns.heatmap(imcoh2[i]-imcoh1[i],xticklabels=imcoh1[i].index,yticklabels=imcoh1[i].columns,mask=imcoh1[i].isnull(),cmap='bwr',ax=axes[i,2],vmin=-1,vmax=1)
                                axes[i,2].set_ylabel('Electrodes')
                                axes[i,2].set_xlabel('Electrodes')
                                axes[i,2].figure.axes[-1].set_ylabel('Difference of\nimaginary coherence values')
                            plt.show()
                        elif len(event_dict.keys())==2:
                            fig,axes = plt.subplots(nrows=3,ncols=3)
                            rows=list(event_dict.keys())+['Difference\n'+list(event_dict.keys())[1]+'\n'+list(event_dict.keys())[0]]
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            pad=5
                            for ax, row in zip(axes[:,0], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for ax, col in zip(axes[0,:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            for i in range(3):
                                if i<2:
                                    sns.heatmap(imcoh1[i],xticklabels=imcoh1[i].index,yticklabels=imcoh1[i].columns,mask=imcoh1[i].isnull(),cmap='bwr',ax=axes[i,0],vmin=-1,vmax=1)
                                else:
                                    sns.heatmap(imcoh1[1]-imcoh1[0],xticklabels=imcoh1[0].index,yticklabels=imcoh1[0].columns,mask=imcoh1[0].isnull(),cmap='bwr',ax=axes[i,0],vmin=-1,vmax=1)
                                axes[i,0].set_ylabel('Electrodes')
                                axes[i,0].set_xlabel('Electrodes')
                                if i<2:
                                    axes[i,0].figure.axes[-1].set_ylabel('Imaginary coherence')
                                else:
                                    axes[i,0].figure.axes[-1].set_ylabel('Difference of\nimaginary coherence values')
                                if i<2:
                                    sns.heatmap(imcoh2[i],xticklabels=imcoh2[i].index,yticklabels=imcoh2[i].columns,mask=imcoh2[i].isnull(),cmap='bwr',ax=axes[i,1],vmin=-1,vmax=1)
                                else:
                                    sns.heatmap(imcoh2[1]-imcoh2[0],xticklabels=imcoh2[0].index,yticklabels=imcoh2[0].columns,mask=imcoh2[0].isnull(),cmap='bwr',ax=axes[i,1],vmin=-1,vmax=1)
                                axes[i,1].set_ylabel('Electrodes')
                                axes[i,1].set_xlabel('Electrodes')
                                if i<2:
                                    axes[i,1].figure.axes[-1].set_ylabel('Imaginary coherence')
                                else:
                                    axes[i,1].figure.axes[-1].set_ylabel('Difference of\nimaginary coherence values')
                                if i<2:
                                    sns.heatmap(imcoh2[i]-imcoh1[i],xticklabels=imcoh1[i].index,yticklabels=imcoh1[i].columns,mask=imcoh1[i].isnull(),cmap='bwr',ax=axes[i,2],vmin=-1,vmax=1)
                                    axes[i,2].set_ylabel('Electrodes')
                                    axes[i,2].set_xlabel('Electrodes')
                                    axes[i,2].figure.axes[-1].set_ylabel('Difference of\nimaginary coherence values')
                            fig.delaxes(axes[2,2])
                            plt.show()
                Button(win,text="Save results",command=save_corr).grid(row=4,column=0,padx=10,sticky=W)
                Button(win,text="Plot coherence",command=plot_coh).grid(row=5,column=0,padx=10,sticky=W)
                Button(win,text="Plot imaginary coherence",command=plot_imcoh).grid(row=6,column=0,padx=10,sticky=W)
                Button(win,text="Close",command=win.destroy).grid(row=7,column=0,padx=10,pady=10,columnspan=3)
            else:
                showinfo(title="Error",message="Frequency must be a positive number greater than zero\n and maximum frequency must be greater than minumum frequency")
        btn=Button(win, text='Start calculation', command=step)
        btn.grid(row=3,column=0,sticky=W,padx=10)
        pbar.grid(row=3,column=1,sticky=W)
        pbtxt.grid(row=3,column=2,sticky=W)

#weighted phase lag index and comparison between conditions
def wpli():
    win=Toplevel(main)
    global x1
    global x2
    fmax=StringVar()
    fmax.set("12")
    fmin=StringVar()
    fmin.set("10")
    error=make_x()
    if error==1:
        Label(win,text="ERROR\nIt is required at least\n1 preprocessed data",justify=CENTER).grid(row=0,column=0,padx=10,pady=10)        
        Button(win,text="OK",command=win.destroy).grid(row=1,column=0,padx=10)
    else:
        n_chans=len(eeg1.ch_names)
        Label(win,text="Calculate correlations").grid(row=0,column=0,columnspan=3,padx=10,pady=10)
        Label(win,text="Minimum frequency (Hz):").grid(row=1,column=0,padx=10,sticky=W)
        Entry(win,textvariable=fmin,width=4).grid(row=1,column=1,sticky=W)
        Label(win,text="Maximum frequency (Hz):").grid(row=2,column=0,padx=10,sticky=W)
        Entry(win,textvariable=fmax,width=4).grid(row=2,column=1,sticky=W)
        pbar=Progressbar(win,orient=HORIZONTAL,length=100,mode='determinate')
        pbar['value']=0.0
        pbtxt=Label(win,text="--")
        def step():
            if (float(fmin.get())>0) and (float(fmax.get())>=float(fmin.get())):
                n_cond=1
                if x2 is not None:
                    n_cond=2
                pbar['value']=0.0
                k=0
                key_idx=0
                pli1=[0 for i in event_dict.keys()]
                if x2 is not None:
                    pli2=[0 for i in event_dict.keys()]
                for key in event_dict.keys(): 
                    win.update_idletasks()
                    win.update()
                    k+=1
                    pbar['value'] += 100/(len(event_dict.keys())*n_cond)
                    pbtxt['text']=f"{k:d}/{len(event_dict.keys())*n_cond:d}"
                    a=mne_connectivity.spectral_connectivity_epochs(x1[key],indices=None,method='wpli',mode='multitaper',sfreq=x1.info['sfreq'],
                                                                     fmin=float(fmin.get()),fmax=float(fmax.get()),faverage=True,mt_adaptive=False,n_jobs=1,verbose='ERROR')
                    pli1[key_idx]=np.reshape(a.get_data(),(n_chans,n_chans))
                    pli1[key_idx][:,:]=pli1[key_idx][:,:]+pli1[key_idx][:,:].T
                    if x2 is not None:
                        win.update_idletasks()
                        win.update()
                        k+=1
                        pbar['value'] += 100/(len(event_dict.keys())*n_cond)
                        pbtxt['text']=f"{k:d}/{len(event_dict.keys())*n_cond:d}"
                        b=mne_connectivity.spectral_connectivity_epochs(x2[key],indices=None,method='wpli',mode='multitaper',sfreq=x2.info['sfreq'],
                                                                     fmin=float(fmin.get()),fmax=float(fmax.get()),faverage=True,mt_adaptive=False,n_jobs=1,verbose='ERROR')
                        pli2[key_idx]=np.reshape(b.get_data(),(n_chans,n_chans))
                        pli2[key_idx][:,:]=pli2[key_idx][:,:]+pli2[key_idx][:,:].T
                    key_idx+=1
                pli1=[pd.DataFrame(pli1[key_idx][:,:],columns=x1.ch_names,index=x1.ch_names) for key_idx in range(len(event_dict.keys()))]
                if x2 is not None:
                    pli2=[pd.DataFrame(pli2[key_idx][:,:],columns=x2.ch_names,index=x2.ch_names) for key_idx in range(len(event_dict.keys()))]                
                def save_corr():
                    showinfo(title="Info",message="You'll save many files,\n one for the calculated weighted PLI \n for each condition/event (CSV tables)")
                    key_idx=0
                    for key in event_dict.keys():
                        #save coh
                        fname1 = fd.asksaveasfilename(title="Weighted PLI for condition "+cond1_name.get()+" event "+key,defaultextension=".csv",filetypes=(("Comma Separated Values", "*.csv"),("All Files", "*.*")))
                        df=pd.DataFrame(pli1[key_idx],index=x1.ch_names,columns=x1.ch_names)
                        df.to_csv(fname1)
                        if x2 is not None:
                            fname2 = fd.asksaveasfilename(title="Weighted PLI for condition "+cond2_name.get()+" event "+key,defaultextension=".csv",filetypes=(("Comma Separated Values", "*.csv"),("All Files", "*.*")))
                            df2=pd.DataFrame(pli2[key_idx],index=x2.ch_names,columns=x2.ch_names)
                            df2.to_csv(fname2)
                        key_idx+=1
                def plot_pli():
                    if x2 is None:
                        if len(event_dict.keys())==1:
                            fig, axes = plt.subplots(nrows=1, ncols=1)
                            sns.heatmap(pli1[0],xticklabels=pli1[0].index,yticklabels=pli1[0].columns,mask=pli1[0].isnull(),cmap='Reds',ax=axes,vmin=0,vmax=1)
                            plt.xlabel("Electrodes")
                            plt.ylabel("Electrodes")
                            axes.figure.axes[-1].set_ylabel('Weighted PLI',size=14)
                            plt.show()
                        elif len(event_dict.keys())>2:
                            fig,axes = plt.subplots(nrows=len(event_dict.keys()))
                            rows=list(event_dict.keys())
                            pad=5
                            for ax, row in zip(axes[:], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for i in range(len(event_dict.keys())):
                                sns.heatmap(pli1[i],xticklabels=pli1[i].index,yticklabels=pli1[i].columns,mask=pli1[i].isnull(),cmap='Reds',ax=axes[i],vmin=0,vmax=1)
                                axes[i].set_ylabel('Electrodes')
                                axes[i].set_xlabel('Electrodes')
                                axes[i].figure.axes[-1].set_ylabel('Weighted PLI')
                            plt.show()
                        elif len(event_dict.keys())==2:
                            fig,axes = plt.subplots(nrows=3)
                            rows=list(event_dict.keys())+['Difference\n'+list(event_dict.keys())[1]+'\n'+list(event_dict.keys())[0]]
                            pad=5
                            for ax, row in zip(axes[:], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for i in range(len(rows)):
                                if i==len(rows)-1:
                                    sns.heatmap(pli1[1]-pli1[0],xticklabels=pli1[0].index,yticklabels=pli1[0].columns,mask=pli1[0].isnull(),cmap='bwr',ax=axes[i],vmin=-1,vmax=1)
                                else:
                                    sns.heatmap(pli1[i],xticklabels=pli1[i].index,yticklabels=pli1[i].columns,mask=pli1[i].isnull(),cmap='Reds',ax=axes[i],vmin=0,vmax=1)
                                axes[i].set_ylabel('Electrodes')
                                axes[i].set_xlabel('Electrodes')
                                if i==len(rows)-1:
                                    axes[i].figure.axes[-1].set_ylabel('Difference of\nWeighted PLI values')
                                else:
                                    axes[i].figure.axes[-1].set_ylabel('Weighted PLI')
                            plt.show()
                    else:
                        if len(event_dict.keys())==1:
                            fig, axes = plt.subplots(nrows=1, ncols=3)
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            pad=5
                            for ax, col in zip(axes[:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            sns.heatmap(pli1[0],xticklabels=pli1[0].index,yticklabels=pli1[0].columns,mask=pli1[0].isnull(),cmap='Reds',ax=axes[0],vmin=0,vmax=1)
                            axes[0].set_ylabel('Electrodes')
                            axes[0].set_xlabel('Electrodes')
                            axes[0].figure.axes[-1].set_ylabel('Weighted PLI')
                            sns.heatmap(pli2[0],xticklabels=pli2[0].index,yticklabels=pli2[0].columns,mask=pli2[0].isnull(),cmap='Reds',ax=axes[1],vmin=0,vmax=1)
                            axes[1].set_ylabel('Electrodes')
                            axes[1].set_xlabel('Electrodes')
                            axes[1].figure.axes[-1].set_ylabel('Weighted PLI')
                            sns.heatmap(pli2[0]-pli1[0],xticklabels=pli1[0].index,yticklabels=pli1[0].columns,mask=pli1[0].isnull(),cmap='bwr',ax=axes[2],vmin=-1,vmax=1)
                            axes[2].set_ylabel('Electrodes')
                            axes[2].set_xlabel('Electrodes')
                            axes[2].figure.axes[-1].set_ylabel('Difference of\nWeighted PLI values')
                            plt.show()
                        elif len(event_dict.keys())>2:
                            fig,axes = plt.subplots(nrows=len(event_dict.keys()),ncols=3)
                            rows=list(event_dict.keys())
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            pad=5
                            for ax, row in zip(axes[:,0], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',size=17,ha='right',va='center')
                            for ax, col in zip(axes[0,:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            for i in range(len(event_dict.keys())):
                                sns.heatmap(pli1[i],xticklabels=pli1[i].index,yticklabels=pli1[i].columns,mask=pli1[i].isnull(),cmap='Reds',ax=axes[i,0],vmin=0,vmax=1)
                                axes[i,0].set_ylabel('Electrodes')
                                axes[i,0].set_xlabel('Electrodes')
                                axes[i,0].figure.axes[-1].set_ylabel('Weighted PLI')
                                sns.heatmap(pli2[i],xticklabels=pli2[i].index,yticklabels=pli2[i].columns,mask=pli2[i].isnull(),cmap='Reds',ax=axes[i,1],vmin=0,vmax=1)
                                axes[i,1].set_ylabel('Electrodes')
                                axes[i,1].set_xlabel('Electrodes')
                                axes[i,1].figure.axes[-1].set_ylabel('Weighted PLI')
                                sns.heatmap(pli2[i]-pli1[i],xticklabels=pli1[i].index,yticklabels=pli1[i].columns,mask=pli1[i].isnull(),cmap='bwr',ax=axes[i,2],vmin=-1,vmax=1)
                                axes[i,2].set_ylabel('Electrodes')
                                axes[i,2].set_xlabel('Electrodes')
                                axes[i,2].figure.axes[-1].set_ylabel('Difference of\nWeighted PLI values')
                            plt.show()
                        elif len(event_dict.keys())==2:
                            fig,axes = plt.subplots(nrows=3,ncols=3)
                            rows=list(event_dict.keys())+['Difference\n'+list(event_dict.keys())[1]+'\n'+list(event_dict.keys())[0]]
                            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
                            pad=5
                            for ax, row in zip(axes[:,0], rows):
                                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
                            for ax, col in zip(axes[0,:], cols):
                                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
                            for i in range(3):
                                if i<2:
                                    sns.heatmap(pli1[i],xticklabels=pli1[i].index,yticklabels=pli1[i].columns,mask=pli1[i].isnull(),cmap='Reds',ax=axes[i,0],vmin=0,vmax=1)
                                else:
                                    sns.heatmap(pli1[1]-pli1[0],xticklabels=pli1[0].index,yticklabels=pli1[0].columns,mask=pli1[0].isnull(),cmap='bwr',ax=axes[i,0],vmin=-1,vmax=1)
                                axes[i,0].set_ylabel('Electrodes')
                                axes[i,0].set_xlabel('Electrodes')
                                if i<2:
                                    axes[i,0].figure.axes[-1].set_ylabel('Weighted PLI')
                                else:
                                    axes[i,0].figure.axes[-1].set_ylabel('Difference of\nWeighted PLI values')
                                if i<2:
                                    sns.heatmap(pli2[i],xticklabels=pli2[i].index,yticklabels=pli2[i].columns,mask=pli2[i].isnull(),cmap='Reds',ax=axes[i,1],vmin=0,vmax=1)
                                else:
                                    sns.heatmap(pli2[1]-pli2[0],xticklabels=pli2[0].index,yticklabels=pli2[0].columns,mask=pli2[0].isnull(),cmap='bwr',ax=axes[i,1],vmin=-1,vmax=1)
                                axes[i,1].set_ylabel('Electrodes')
                                axes[i,1].set_xlabel('Electrodes')
                                if i<2:
                                    axes[i,1].figure.axes[-1].set_ylabel('Weighted PLI')
                                else:
                                    axes[i,1].figure.axes[-1].set_ylabel('Difference of\nWeighted PLI values')
                                if i<2:
                                    sns.heatmap(pli2[i]-pli1[i],xticklabels=pli1[i].index,yticklabels=pli1[i].columns,mask=pli1[i].isnull(),cmap='bwr',ax=axes[i,2],vmin=-1,vmax=1)
                                    axes[i,2].set_ylabel('Electrodes')
                                    axes[i,2].set_xlabel('Electrodes')
                                    axes[i,2].figure.axes[-1].set_ylabel('Difference of\nWeighted PLI values')
                            fig.delaxes(axes[2,2])
                            plt.show()
                Button(win,text="Save results",command=save_corr).grid(row=4,column=0,padx=10,sticky=W)
                Button(win,text="Plot weighted PLI",command=plot_pli).grid(row=5,column=0,padx=10,sticky=W)
                Button(win,text="Close",command=win.destroy).grid(row=7,column=0,padx=10,pady=10,columnspan=3)
            else:
                showinfo(title="Error",message="Frequency must be a positive number greater than zero\n and maximum frequency must be greater than minimum frequency")
        btn=Button(win, text='Start calculation', command=step)
        btn.grid(row=3,column=0,sticky=W,padx=10)
        pbar.grid(row=3,column=1,sticky=W)
        pbtxt.grid(row=3,column=2,sticky=W)

#makes the frames for the videos of dynamic functional connectivity (comparative case when calculation based on average of epochs)
def make_frame(y1,y2=None,vmin=-1,vmax=1,color='bwr',lblttl='Pearson correlation',frttl='my_frame.png'):
    ''' y1    = list of dataframes for condition 1.
                Each list index correponds to an event.
                Each dataframe has dimensions of (n_chans,n_chans),
                with cell values corresponding to instant
                correlation/mutual information/transfer entropy
                between the two channels.
        y2    = idem, for condition 2.
        vmin  = minimum value for colorbar
        vmax  = maximum value for colorbar
        color = color scheme for colorbar
        lblttl= colorbar label title (the connectivity measure used)
        frttl = frame title'''
    if y2 is None:
        if len(event_dict.keys())==1:
            fig, axes = plt.subplots(nrows=1, ncols=1)
            sns.heatmap(y1[0],xticklabels=y1[0].index,yticklabels=y1[0].columns,mask=y1[0].isnull(),cmap=color,ax=axes,vmin=vmin,vmax=vmax)
            plt.xlabel("Electrodes")
            plt.ylabel("Electrodes")
            axes.figure.axes[-1].set_ylabel(lblttl)
            fig.savefig(frttl,dpi=300)
            fig.clf()
            plt.close()
        elif len(event_dict.keys())>2:
            fig,axes = plt.subplots(nrows=len(event_dict.keys()))
            rows=list(event_dict.keys())
            pad=5
            for ax, row in zip(axes[:], rows):
                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
            for i in range(len(event_dict.keys())):
                sns.heatmap(y1[i],xticklabels=y1[i].index,yticklabels=y1[i].columns,mask=y1[i].isnull(),cmap=color,ax=axes[i],vmin=vmin,vmax=vmax)
                axes[i].set_ylabel('Electrodes')
                axes[i].set_xlabel('Electrodes')
                axes[i].figure.axes[-1].set_ylabel(lblttl)
            fig.savefig(frttl,dpi=300)
            fig.clf()
            plt.close()
        elif len(event_dict.keys())==2:
            fig,axes = plt.subplots(nrows=3)
            rows=list(event_dict.keys())+['Difference\n'+list(event_dict.keys())[1]+'\n'+list(event_dict.keys())[0]]
            pad=5
            for ax, row in zip(axes[:], rows):
                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center')
            for i in range(len(rows)):
                if i==len(rows)-1:
                    diff=y1[1]-y1[0]
                    minbar=min[diff.min().min(),-diff.max().max()]
                    maxbar=max[diff.max().max(),-diff.min().min()]
                    sns.heatmap(diff,xticklabels=diff.index,yticklabels=diff.columns,mask=diff.isnull(),cmap='bwr',ax=axes[i],vmin=minbar,vmax=maxbar)
                else:
                    sns.heatmap(y1[i],xticklabels=y1[i].index,yticklabels=y1[i].columns,mask=y1[i].isnull(),cmap=color,ax=axes[i],vmin=vmin,vmax=vmax)
                axes[i].set_ylabel('Electrodes')
                axes[i].set_xlabel('Electrodes')
                if i==len(rows)-1:
                    axes[i].figure.axes[-1].set_ylabel(f'Difference of\n{lblttl} values')
                else:
                    axes[i].figure.axes[-1].set_ylabel(lblttl)
            fig.savefig(frttl,dpi=300)
            fig.clf()
            plt.close()
    else:
        if len(event_dict.keys())==1:
            fig, axes = plt.subplots(nrows=1, ncols=3)
            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
            pad=5
            for ax, col in zip(axes[:], cols):
                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
            sns.heatmap(y1[0],xticklabels=y1[0].index,yticklabels=y1[0].columns,mask=y1[0].isnull(),cmap=color,ax=axes[0],vmin=vmin,vmax=vmax)
            axes[0].set_ylabel('Electrodes')
            axes[0].set_xlabel('Electrodes')
            axes[0].figure.axes[-1].set_ylabel(lblttl)
            sns.heatmap(y2[0],xticklabels=y2[0].index,yticklabels=y2[0].columns,mask=y2[0].isnull(),cmap=color,ax=axes[1],vmin=vmin,vmax=vmax)
            axes[1].set_ylabel('Electrodes')
            axes[1].set_xlabel('Electrodes')
            axes[1].figure.axes[-1].set_ylabel(lblttl)
            diff=y2[0]-y1[0]
            minbar=min[diff.min().min(),-diff.max().max()]
            maxbar=max[diff.max().max(),-diff.min().min()]
            sns.heatmap(diff,xticklabels=diff.index,yticklabels=diff.columns,mask=diff.isnull(),cmap='bwr',ax=axes[2],vmin=minbar,vmax=maxbar)
            axes[2].set_ylabel('Electrodes')
            axes[2].set_xlabel('Electrodes')
            axes[2].figure.axes[-1].set_ylabel(f'Difference of\n{lblttl} values')
            fig.savefig(frttl,dpi=300)
            fig.clf()
            plt.close()
        elif len(event_dict.keys())>2:
            fig,axes = plt.subplots(nrows=len(event_dict.keys()),ncols=3)
            rows=list(event_dict.keys())
            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
            pad=5
            for ax, row in zip(axes[:,0], rows):
                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',size=17,ha='right',va='center')
            for ax, col in zip(axes[0,:], cols):
                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline')
            tempmin=[]
            tempmax=[]
            for i in range(len(event_dict.keys())):
                tempdiff=y2[i]-y1[i]
                tempmin.append(tempdiff.min().min())
                tempmax.append(tempmax.max().max())
            minbar=min(min(tempmin),-max(tempmax))
            maxbar=max(max(tempmax),-min(tempmin))
            for i in range(len(event_dict.keys())):
                sns.heatmap(y1[i],xticklabels=y1[i].index,yticklabels=y1[i].columns,mask=y1[i].isnull(),cmap=color,ax=axes[i,0],vmin=vmin,vmax=vmax)
                axes[i,0].set_ylabel('Electrodes')
                axes[i,0].set_xlabel('Electrodes')
                axes[i,0].figure.axes[-1].set_ylabel(lblttl)
                sns.heatmap(y2[i],xticklabels=y2[i].index,yticklabels=y2[i].columns,mask=y2[i].isnull(),cmap=color,ax=axes[i,1],vmin=vmin,vmax=vmax)
                axes[i,1].set_ylabel('Electrodes')
                axes[i,1].set_xlabel('Electrodes')
                axes[i,1].figure.axes[-1].set_ylabel(lblttl)
                sns.heatmap(y2[i]-y1[i],xticklabels=y1[i].index,yticklabels=y1[i].columns,mask=y1[i].isnull(),cmap='bwr',ax=axes[i,2],vmin=minbar,vmax=maxbar)
                axes[i,2].set_ylabel('Electrodes')
                axes[i,2].set_xlabel('Electrodes')
                axes[i,2].figure.axes[-1].set_ylabel(f'Difference of\n{lblttl} values')
            fig.savefig(frttl,dpi=300)
            fig.clf()
            plt.close()
        elif len(event_dict.keys())==2:
            fig,axes = plt.subplots(nrows=3,ncols=3)
            plt.subplots_adjust(left=0.18,wspace=0.6,hspace=0.4)
            rows=list(event_dict.keys())+['Difference\n'+list(event_dict.keys())[1]+'\n'+list(event_dict.keys())[0]]
            cols=[cond1_name.get(),cond2_name.get()]+['Difference\n'+cond2_name.get()+'\n'+cond1_name.get()]
            pad=5
            for ax, row in zip(axes[:,0], rows):
                ax.annotate(row,xy=(0, 0.5),xytext=(-ax.yaxis.labelpad - pad, 0),xycoords=ax.yaxis.label,textcoords='offset points',ha='right',va='center',fontsize=6)
            for ax, col in zip(axes[0,:], cols):
                ax.annotate(col,xy=(0.5,1),xytext=(0,pad),xycoords='axes fraction',textcoords='offset points',ha='center',va='baseline',fontsize=6)
            tempmax=[]
            tempmin=[]
            tempdiff1=y1[1]-y1[0]
            tempdiff2=y2[1]-y2[0]
            tempmax.append(tempdiff1.max().max())
            tempmin.append(tempdiff1.min().min())
            tempmax.append(tempdiff2.max().max())
            tempmin.append(tempdiff2.min().min())
            for i in range(2):
                tempdiff3=y2[i]-y1[i]
                tempmax.append(tempdiff3.max().max())
                tempmin.append(tempdiff3.min().min())
            minbar=min(min(tempmin),-max(tempmax))
            maxbar=max(max(tempmax),-min(tempmin))                
            for i in range(3):
                if i<2:
                    sns.heatmap(y1[i],xticklabels=y1[i].index,yticklabels=y1[i].columns,mask=y1[i].isnull(),cmap=color,ax=axes[i,0],vmin=vmin,vmax=vmax)
                else:
                    sns.heatmap(y1[1]-y1[0],xticklabels=y1[0].index,yticklabels=y1[0].columns,mask=y1[0].isnull(),cmap='bwr',ax=axes[i,0],vmin=minbar,vmax=maxbar)
                axes[i,0].set_ylabel('Electrodes',fontsize=4)
                axes[i,0].set_xlabel('Electrodes',fontsize=4)
                axes[i,0].tick_params(axis='both', which='major', labelsize=4)
                cbar = axes[i,0].collections[0].colorbar
                cbar.ax.tick_params(labelsize=4)
                if i<2:
                    axes[i,0].figure.axes[-1].set_ylabel(lblttl,fontsize=4)
                else:
                    axes[i,0].figure.axes[-1].set_ylabel(f'Difference of\n{lblttl} values',fontsize=4)
                if i<2:
                    sns.heatmap(y2[i],xticklabels=y2[i].index,yticklabels=y2[i].columns,mask=y2[i].isnull(),cmap=color,ax=axes[i,1],vmin=vmin,vmax=vmax)
                else:
                    sns.heatmap(y2[1]-y2[0],xticklabels=y2[0].index,yticklabels=y2[0].columns,mask=y2[0].isnull(),cmap='bwr',ax=axes[i,1],vmin=minbar,vmax=maxbar)
                axes[i,1].set_ylabel('Electrodes',fontsize=4)
                axes[i,1].set_xlabel('Electrodes',fontsize=4)
                axes[i,1].tick_params(axis='both', which='major', labelsize=4)
                cbar = axes[i,1].collections[0].colorbar
                cbar.ax.tick_params(labelsize=4)
                if i<2:
                    axes[i,1].figure.axes[-1].set_ylabel(lblttl,fontsize=4)
                else:
                    axes[i,1].figure.axes[-1].set_ylabel(f'Difference of\n{lblttl} values',fontsize=4)
                if i<2:
                    sns.heatmap(y2[i]-y1[i],xticklabels=y1[i].index,yticklabels=y1[i].columns,mask=y1[i].isnull(),cmap='bwr',ax=axes[i,2],vmin=minbar,vmax=maxbar)
                    axes[i,2].set_ylabel('Electrodes',fontsize=4)
                    axes[i,2].set_xlabel('Electrodes',fontsize=4)
                    axes[i,2].tick_params(axis='both', which='major', labelsize=4)
                    axes[i,2].figure.axes[-1].set_ylabel(f'Difference of\n{lblttl} values',fontsize=4)
                    cbar = axes[i,2].collections[0].colorbar
                    cbar.ax.tick_params(labelsize=4)
            fig.delaxes(axes[2,2])
            fig.savefig(frttl,dpi=300)
            fig.clf()
            plt.close()

#makes the frames for the videos of dynamic functional connectivity (simpler case when calculation based on raw or just 1 epoch)
def make_frame2(y,vmin=-1,vmax=1,color='bwr',lblttl='Pearson correlation',frttl='my_frame.png'):
    ''' y     = dataframe for eeg data
                dimensions of (n_chans,n_chans),
                with cell values corresponding to instant
                correlation/mutual information/transfer entropy
                between the two channels.
        vmin  = minimum value for colorbar
        vmax  = maximum value for colorbar
        color = color scheme for colorbar
        lblttl= colorbar label title (the connectivity measure used)
        frttl = frame title'''
    fig, axes = plt.subplots(nrows=1, ncols=1)
    sns.heatmap(y,xticklabels=y.index,yticklabels=y.columns,mask=y.isnull(),cmap=color,ax=axes,vmin=vmin,vmax=vmax)
    plt.xlabel("Electrodes")
    plt.ylabel("Electrodes")
    axes.figure.axes[-1].set_ylabel(lblttl)
    fig.savefig(frttl,dpi=300)
    fig.clf()
    plt.close()
    
#pearson correlation based dynamic functional connectivity
def pearson_dfc():
    win=Toplevel(main)
    raw_or_epoch=IntVar()
    raw_or_epoch.set(3)
    delay=StringVar()
    delay.set("10")
    wlen=StringVar()
    wlen.set("500")
    woverlap=StringVar()
    woverlap.set("250")
    sel_epoch1=StringVar()
    sel_epoch1.set("1")
    sel_epoch2=StringVar()
    sel_epoch2.set("1")
    fr=StringVar()
    fr.set("5")
    error=make_x()
    if (error==1) and (raw1 is None):
        Label(win,text="ERROR\nIt is required at least\n1 preprocessed or raw data",justify=CENTER).grid(row=0,column=0,padx=10,pady=10)        
        Button(win,text="OK",command=win.destroy).grid(row=1,column=0,padx=10)
    else:
        def step():
            if (int(delay.get())>=0) and (int(wlen.get())>0) and (int(fr.get())>0):
                error2=0
                if (raw_or_epoch.get()==2) and (error==1):
                    showinfo(title="Error",message="To work with epochs it is\nnecessary to have at\nleast 1 preprocessed data")
                    error2=1
                elif (raw_or_epoch.get()==1) and (raw1 is None):
                    showinfo(title="Error",message="To work with raw it is\nnecessary to have loaded at\nleast 1 raw EDF file")
                    error2=1
                elif raw_or_epoch.get()==1:
                    n_chans=len(xraw1.ch_names)
                    delayval=int(float(delay.get())*xraw1.info['sfreq']/1000)
                    wlenval=int(float(wlen.get())*xraw1.info['sfreq']/1000)
                    woverlapval=int(float(woverlap.get())*xraw1.info['sfreq']/1000)
                    vals1=xraw1.get_data()
                    starttimes1=np.arange(0,vals1.shape[1]-wlenval-delayval,wlenval-woverlapval)
                    endtimes1=starttimes1+wlenval
                    dfc1=np.empty((len(starttimes1),n_chans,n_chans))
                    dfc1.fill(np.nan)
                    if xraw2 is not None:
                        vals2=xraw2.get_data()
                        starttimes2=np.arange(0,vals2.shape[1]-wlenval-delayval,wlenval-woverlapval)
                        endtimes2=starttimes2+wlenval
                        dfc2=np.empty((len(starttimes2),n_chans,n_chans))
                        dfc2.fill(np.nan)
                    pbar['value']=0.0
                    maxrounds=0
                    for i in range(n_chans):
                        for j in range(n_chans):
                            for m1 in range(len(starttimes1)):
                                maxrounds+=1
                            if xraw2 is not None:
                                for m2 in range(len(starttimes2)):
                                    maxrounds+=1
                    k=0                
                    for i in range(n_chans):
                        for j in range(n_chans):
                            for m1 in range(len(starttimes1)):
                                win.update_idletasks()
                                win.update()
                                k+=1
                                pbar['value'] += 100/maxrounds
                                pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                key_idx=0
                                dfc1[m1,i,j]=np.corrcoef(vals1[i,starttimes1[m1]:endtimes1[m1]],vals1[j,starttimes1[m1]+delayval:endtimes1[m1]+delayval])[0,1]
                            if xraw2 is not None:
                                for m2 in range(len(starttimes2)):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar['value'] += 100/maxrounds
                                    pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                    key_idx=0
                                    dfc2[m2,i,j]=np.corrcoef(vals2[i,starttimes2[m2]:endtimes2[m2]],vals2[j,starttimes2[m2]+delayval:endtimes2[m2]+delayval])[0,1]                                    
                elif raw_or_epoch.get()==2:
                    n_chans=len(x1.ch_names)
                    delayval=int(float(delay.get())*x1.info['sfreq']/1000)
                    wlenval=int(float(wlen.get())*x1.info['sfreq']/1000)
                    woverlapval=int(float(woverlap.get())*x1.info['sfreq']/1000)
                    vals1=x1.get_data()[int(sel_epoch1.get()),:,:]
                    starttimes1=np.arange(0,vals1.shape[1]-wlenval-delayval,wlenval-woverlapval)
                    endtimes1=starttimes1+wlenval
                    dfc1=np.empty((len(starttimes1),n_chans,n_chans))
                    dfc1.fill(np.nan)
                    if x2 is not None:
                        vals2=x2.get_data()[int(sel_epoch2.get()),:,:]
                        starttimes2=np.arange(0,vals2.shape[1]-wlenval-delayval,wlenval-woverlapval)
                        endtimes2=starttimes2+wlenval
                        dfc2=np.empty((len(starttimes2),n_chans,n_chans))
                        dfc2.fill(np.nan)
                    pbar['value']=0.0
                    maxrounds=0
                    for i in range(n_chans):
                        for j in range(n_chans):
                            for m1 in range(len(starttimes1)):
                                maxrounds+=1
                            if x2 is not None:
                                for m2 in range(len(starttimes2)):
                                    maxrounds+=1
                    k=0                
                    for i in range(n_chans):
                        for j in range(n_chans):
                            for m1 in range(len(starttimes1)):
                                win.update_idletasks()
                                win.update()
                                k+=1
                                pbar['value'] += 100/maxrounds
                                pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                key_idx=0
                                dfc1[m1,i,j]=np.corrcoef(vals1[i,starttimes1[m1]:endtimes1[m1]],vals1[j,starttimes1[m1]+delayval:endtimes1[m1]+delayval])[0,1]
                            if x2 is not None:
                                for m2 in range(len(starttimes2)):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar['value'] += 100/maxrounds
                                    pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                    key_idx=0
                                    dfc2[m2,i,j]=np.corrcoef(vals2[i,starttimes2[m2]:endtimes2[m2]],vals2[j,starttimes2[m2]+delayval:endtimes2[m2]+delayval])[0,1]                                    
                elif raw_or_epoch.get()==3:
                    event_names=list(event_dict.keys())
                    n_chans=len(x1.ch_names)
                    delayval=int(float(delay.get())*x1.info['sfreq']/1000)
                    wlenval=int(float(wlen.get())*x1.info['sfreq']/1000)
                    woverlapval=int(float(woverlap.get())*x1.info['sfreq']/1000)
                    vals1=[]
                    n_epochs1=[]
                    for i in range(len(event_names)):
                        vals1.append(x1[event_names[i]].get_data())
                        n_epochs1.append(vals1[i].shape[0])
                    starttimes1=np.arange(0,vals1[0].shape[2]-wlenval-delayval,wlenval-woverlapval)
                    endtimes1=starttimes1+wlenval
                    tdfc1=[]
                    for i in range(len(event_names)):
                        tdfc1.append(np.empty((n_epochs1[i],len(starttimes1),n_chans,n_chans)))
                        tdfc1[i].fill(np.nan)
                    dfc1=np.empty((len(event_names),len(starttimes1),n_chans,n_chans))
                    dfc1.fill(np.nan)
                    if x2 is not None:
                        vals2=[]
                        n_epochs2=[]
                        for i in range(len(event_names)):
                            vals2.append(x2[event_names[i]].get_data())
                            n_epochs2.append(vals2[i].shape[0])
                        starttimes2=np.arange(0,vals2[0].shape[2]-wlenval-delayval,wlenval-woverlapval)
                        endtimes2=starttimes2+wlenval
                        tdfc2=[]
                        for i in range(len(event_names)):
                            tdfc2.append(np.empty((n_epochs2[i],len(starttimes1),n_chans,n_chans)))
                            tdfc2[i].fill(np.nan)
                        dfc2=np.empty((len(event_names),len(starttimes2),n_chans,n_chans))
                        dfc2.fill(np.nan)
                    pbar['value']=0.0
                    maxrounds=0
                    for ev in range(len(event_names)):
                        for i in range(n_chans):
                            for j in range(n_chans):
                                for m1 in range(len(starttimes1)):
                                    for e1 in range(n_epochs1[ev]):
                                        maxrounds+=1
                                if x2 is not None:
                                    for m2 in range(len(starttimes2)):
                                        for e2 in range(n_epochs2[ev]):
                                            maxrounds+=1
                    k=0                
                    for ev in range(len(event_names)):
                        for i in range(n_chans):
                            for j in range(n_chans):
                                for m1 in range(len(starttimes1)):
                                    for e1 in range(n_epochs1[ev]):
                                        win.update_idletasks()
                                        win.update()
                                        k+=1
                                        pbar['value'] += 100/maxrounds
                                        pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                        key_idx=0
                                        tdfc1[ev][e1,m1,i,j]=np.corrcoef(vals1[ev][e1,i,int(starttimes1[m1]):int(endtimes1[m1])],vals1[ev][e1,j,int(starttimes1[m1])+delayval:int(endtimes1[m1])+delayval])[0,1]
                                if x2 is not None:
                                    for m2 in range(len(starttimes2)):
                                        for e2 in range(n_epochs2[ev]):
                                            win.update_idletasks()
                                            win.update()
                                            k+=1
                                            pbar['value'] += 100/maxrounds
                                            pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                            key_idx=0
                                            tdfc2[ev][e2,m2,i,j]=np.corrcoef(vals2[ev][e2,i,starttimes2[m2]:endtimes2[m2]],vals2[ev][e2,j,starttimes2[m2]+delayval:endtimes2[m2]+delayval])[0,1]
                    pbtxt['text']="finishing, please wait"
                    for ev in range(len(event_names)):
                        for i in range(n_chans):
                            for j in range(n_chans):
                                for m1 in range(len(starttimes1)):
                                    dfc1[ev,m1,i,j]=np.nanmean(tdfc1[ev][:,m1,i,j])
                                if x2 is not None:
                                    for m2 in range(len(starttimes2)):
                                        dfc2[ev,m2,i,j]=np.nanmean(tdfc2[ev][:,m2,i,j])        
                    pbtxt['text']="done!"
                if error2==0:
                    def make_film():
                        winfilm=Toplevel(win)
                        Label(winfilm,text="Make Dynamic Functional Connectivity animation").grid(row=0,column=0,padx=10,pady=10,columnspan=3)
                        pbar2=Progressbar(winfilm,orient=HORIZONTAL,length=100,mode='determinate')
                        pbar2['value']=0.0
                        pbtxt2=Label(winfilm,text="--")
                        def process_frames():
                            if raw_or_epoch.get()==3:
                                nocond2=0
                                try:
                                    n_frames=min(len(dfc1[0,:,0,0]),len(dfc2[0,:,0,0]))
                                except:
                                    n_frames=len(dfc1[0,:,0,0])
                                    nocond2=1
                                k=0
                                os.mkdir('temp_DFC_frames')
                                for frame in range(n_frames):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar2['value'] += 100/n_frames
                                    pbtxt2['text']=f"{k:d}/{n_frames:d}"
                                    y1=[]
                                    if nocond2==1:
                                        y2=None
                                    else:
                                        y2=[]
                                    for ev in range(len(event_names)):
                                        y1.append(pd.DataFrame(dfc1[ev,frame,:,:],index=x1.ch_names,columns=x1.ch_names))
                                        if nocond2==0:
                                            y2.append(pd.DataFrame(dfc2[ev,frame,:,:],index=x2.ch_names,columns=x2.ch_names))
                                    make_frame(y1,y2,vmin=-1,vmax=1,color='bwr',lblttl='Pearson correlation',frttl=f'temp_DFC_frames/DFC_frame{frame:05d}.png')
                                pbtxt2['text']="Finishing up"
                                command=f'ffmpeg -framerate {fr.get()} -i temp_DFC_frames/DFC_frame%05d.png -pix_fmt yuv420p DFC_Pearson.mp4'
                                os.system(command)
                                shutil.rmtree('temp_DFC_frames')
                                showinfo(title="Info",message="Video of DFC was saved as DFC_Pearson.mp4 at\n"+os.getcwd())
                            else:
                                nocond2=0
                                n_frames1=len(dfc1[:,0,0])
                                total_frames=n_frames1
                                if raw_or_epoch.get()==2:
                                    ch_names=x1.ch_names
                                else:
                                    ch_names=xraw1.ch_names
                                try:
                                    n_frames2=len(dfc2[:,0,0])
                                    total_frames+=n_frames2
                                except:
                                    nocond2=1
                                k=0
                                os.mkdir('temp_DFC_frames')
                                for frame in range(n_frames1):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar2['value'] += 100/total_frames
                                    pbtxt2['text']=f"{k:d}/{total_frames:d}"
                                    y1=pd.DataFrame(dfc1[frame,:,:],index=ch_names,columns=ch_names)
                                    make_frame2(y1,vmin=-1,vmax=1,color='bwr',lblttl='Pearson correlation',frttl=f'temp_DFC_frames/DFC_{cond1_name.get()}_frame{frame:05d}.png')
                                if nocond==0:
                                    for frame in range(n_frames2):
                                        win.update_idletasks()
                                        win.update()
                                        k+=1
                                        pbar2['value'] += 100/total_frames
                                        pbtxt2['text']=f"{k:d}/{total_frames:d}"
                                        y2=pd.DataFrame(dfc2[frame,:,:],index=ch_names,columns=ch_names)
                                        make_frame2(y2,vmin=-1,vmax=1,color='bwr',lblttl='Pearson correlation',frttl=f'temp_DFC_frames/DFC_{cond2_name.get()}_frame{frame:05d}.png')
                                pbtxt2['text']="Finishing up"
                                command=f'ffmpeg -framerate {fr.get()} -i temp_DFC_frames/DFC_{cond1_name.get()}_frame%05d.png -pix_fmt yuv420p DFC_{cond1_name.get()}_Pearson.mp4'
                                os.system(command)
                                if nocond==0:
                                    command=f'ffmpeg -framerate {fr.get()} -i temp_DFC_frames/DFC_{cond2_name.get()}_frame%05d.png -pix_fmt yuv420p DFC_{cond2_name.get()}_Pearson.mp4'
                                    os.system(command)
                                shutil.rmtree('temp_DFC_frames')
                                if nocond==0:
                                    showinfo(title="Info",message="Videos of DFC were saved as\nDFC_{cond1_name.get()}_Pearson.mp4 and DFC_{cond1_name.get()}_Pearson.mp4 at\n"+os.getcwd())
                                else:
                                    showinfo(title="Info",message="Video of DFC were saved as\nDFC_{cond1_name.get()}_Pearson.mp4 at\n"+os.getcwd())
                            winfilm.destroy()
                        btn2=Button(winfilm, text='Create video', command=process_frames)
                        btn2.grid(row=1,column=0,sticky=W,padx=10)
                        pbar2.grid(row=1,column=1,sticky=W)
                        pbtxt2.grid(row=1,column=2,sticky=W)
                    def save_corr():
                        showinfo(title="Info",message="The full results will be saved as a\nNumpy array of size (n_times,n_chans,n_chans))")
                        fname1 = fd.asksaveasfilename(title="Correlation DFC condition "+cond1_name.get(),defaultextension=".npy",filetypes=(("Numpy array", "*.npy"),("All Files", "*.*")))
                        np.save(fname1,dfc1)
                        if (x2 is not None) or (raw2 is not None):
                            fname2 = fd.asksaveasfilename(title="Correlation DFC condition "+cond2_name.get(),defaultextension=".npy",filetypes=(("Numpy array", "*.npy"),("All Files", "*.*")))
                            np.save(fname2,dfc2)
                    Button(win,text="Save results",command=save_corr).grid(row=10,column=0,padx=10,sticky=W)
                    Button(win,text="Make DFC animations",command=make_film).grid(row=11,column=0,padx=10,sticky=W)
                    Button(win,text="Close",command=win.destroy).grid(row=12,column=0,padx=10,pady=10,columnspan=5)
            else:
                showinfo(title="Error",message="Delay, window length and frame rate must be positive numbers")
        Label(win,text="Calculate correlations").grid(row=0,column=0,columnspan=4,padx=10,pady=10)
        Label(win,text="Calculate from:").grid(row=1,column=0,padx=10,sticky=W)
        Radiobutton(win,text="From raw EEG",variable=raw_or_epoch,value=1).grid(row=1,column=1,sticky=W)
        Radiobutton(win,text="From epoch:",variable=raw_or_epoch,value=2).grid(row=2,column=1,sticky=W)
        Entry(win,textvariable=sel_epoch1,width=2).grid(row=2,column=2,sticky=W)
        Label(win,text=f"for {cond1_name.get()}").grid(row=2,column=3,sticky=W)
        Entry(win,textvariable=sel_epoch2,width=2).grid(row=3,column=2,sticky=W)
        Label(win,text=f"for {cond2_name.get()}").grid(row=3,column=3,sticky=W)
        Radiobutton(win,text="Average of results of all epochs",variable=raw_or_epoch,value=3).grid(row=4,column=1,sticky=W,columnspan=3)
        Label(win,text="Transmission delay between brain regions (ms):").grid(row=5,column=0,padx=10,sticky=W)
        Entry(win,textvariable=delay,width=4).grid(row=5,column=1,sticky=W)
        Label(win,text="Moving window length (ms):").grid(row=6,column=0,padx=10,sticky=W)
        Entry(win,textvariable=wlen,width=6).grid(row=6,column=1,sticky=W)
        Label(win,text="Moving window overlap (ms):").grid(row=7,column=0,padx=10,sticky=W)
        Entry(win,textvariable=woverlap,width=6).grid(row=7,column=1,sticky=W)
        Label(win,text="Frame rate:").grid(row=8,column=0,padx=10,sticky=W)
        Entry(win,textvariable=fr,width=6).grid(row=8,column=1,sticky=W)
        pbar=Progressbar(win,orient=HORIZONTAL,length=100,mode='determinate')
        pbar['value']=0.0
        pbtxt=Label(win,text="--")
        btn=Button(win, text='Start calculation', command=step)
        btn.grid(row=9,column=0,sticky=W,padx=10)
        pbar.grid(row=9,column=1,sticky=W)
        pbtxt.grid(row=9,column=2,sticky=W,columnspan=3)

#spearman correlation based dynamic functional connectivity
def spearman_dfc():
    win=Toplevel(main)
    raw_or_epoch=IntVar()
    raw_or_epoch.set(3)
    delay=StringVar()
    delay.set("10")
    wlen=StringVar()
    wlen.set("500")
    woverlap=StringVar()
    woverlap.set("250")
    sel_epoch1=StringVar()
    sel_epoch1.set("1")
    sel_epoch2=StringVar()
    sel_epoch2.set("1")
    fr=StringVar()
    fr.set("5")
    error=make_x()
    if (error==1) and (raw1 is None):
        Label(win,text="ERROR\nIt is required at least\n1 preprocessed or raw data",justify=CENTER).grid(row=0,column=0,padx=10,pady=10)        
        Button(win,text="OK",command=win.destroy).grid(row=1,column=0,padx=10)
    else:
        def step():
            if (int(delay.get())>=0) and (int(wlen.get())>0) and (int(fr.get())>0):
                error2=0
                if (raw_or_epoch.get()==2) and (error==1):
                    showinfo(title="Error",message="To work with epochs it is\nnecessary to have at\nleast 1 preprocessed data")
                    error2=1
                elif (raw_or_epoch.get()==1) and (raw1 is None):
                    showinfo(title="Error",message="To work with raw it is\nnecessary to have loaded at\nleast 1 raw EDF file")
                    error2=1
                elif raw_or_epoch.get()==1:
                    n_chans=len(xraw1.ch_names)
                    delayval=int(float(delay.get())*xraw1.info['sfreq']/1000)
                    wlenval=int(float(wlen.get())*xraw1.info['sfreq']/1000)
                    woverlapval=int(float(woverlap.get())*xraw1.info['sfreq']/1000)
                    vals1=xraw1.get_data()
                    starttimes1=np.arange(0,vals1.shape[1]-wlenval-delayval,wlenval-woverlapval)
                    endtimes1=starttimes1+wlenval
                    dfc1=np.empty((len(starttimes1),n_chans,n_chans))
                    dfc1.fill(np.nan)
                    if xraw2 is not None:
                        vals2=xraw2.get_data()
                        starttimes2=np.arange(0,vals2.shape[1]-wlenval-delayval,wlenval-woverlapval)
                        endtimes2=starttimes2+wlenval
                        dfc2=np.empty((len(starttimes2),n_chans,n_chans))
                        dfc2.fill(np.nan)
                    pbar['value']=0.0
                    maxrounds=0
                    for i in range(n_chans):
                        for j in range(n_chans):
                            for m1 in range(len(starttimes1)):
                                maxrounds+=1
                            if xraw2 is not None:
                                for m2 in range(len(starttimes2)):
                                    maxrounds+=1
                    k=0                
                    for i in range(n_chans):
                        for j in range(n_chans):
                            for m1 in range(len(starttimes1)):
                                win.update_idletasks()
                                win.update()
                                k+=1
                                pbar['value'] += 100/maxrounds
                                pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                key_idx=0
                                dfc1[m1,i,j]=spearmanr(vals1[i,starttimes1[m1]:endtimes1[m1]],vals1[j,starttimes1[m1]+delayval:endtimes1[m1]+delayval]).correlation
                            if xraw2 is not None:
                                for m2 in range(len(starttimes2)):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar['value'] += 100/maxrounds
                                    pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                    key_idx=0
                                    dfc2[m2,i,j]=spearmanr(vals2[i,starttimes2[m2]:endtimes2[m2]],vals2[j,starttimes2[m2]+delayval:endtimes2[m2]+delayval]).correlation                                   
                elif raw_or_epoch.get()==2:
                    n_chans=len(x1.ch_names)
                    delayval=int(float(delay.get())*x1.info['sfreq']/1000)
                    wlenval=int(float(wlen.get())*x1.info['sfreq']/1000)
                    woverlapval=int(float(woverlap.get())*x1.info['sfreq']/1000)
                    vals1=x1.get_data()[int(sel_epoch1.get()),:,:]
                    starttimes1=np.arange(0,vals1.shape[1]-wlenval-delayval,wlenval-woverlapval)
                    endtimes1=starttimes1+wlenval
                    dfc1=np.empty((len(starttimes1),n_chans,n_chans))
                    dfc1.fill(np.nan)
                    if x2 is not None:
                        vals2=x2.get_data()[int(sel_epoch2.get()),:,:]
                        starttimes2=np.arange(0,vals2.shape[1]-wlenval-delayval,wlenval-woverlapval)
                        endtimes2=starttimes2+wlenval
                        dfc2=np.empty((len(starttimes2),n_chans,n_chans))
                        dfc2.fill(np.nan)
                    pbar['value']=0.0
                    maxrounds=0
                    for i in range(n_chans):
                        for j in range(n_chans):
                            for m1 in range(len(starttimes1)):
                                maxrounds+=1
                            if x2 is not None:
                                for m2 in range(len(starttimes2)):
                                    maxrounds+=1
                    k=0                
                    for i in range(n_chans):
                        for j in range(n_chans):
                            for m1 in range(len(starttimes1)):
                                win.update_idletasks()
                                win.update()
                                k+=1
                                pbar['value'] += 100/maxrounds
                                pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                key_idx=0
                                dfc1[m1,i,j]=spearmanr(vals1[i,starttimes1[m1]:endtimes1[m1]],vals1[j,starttimes1[m1]+delayval:endtimes1[m1]+delayval]).correlation
                            if x2 is not None:
                                for m2 in range(len(starttimes2)):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar['value'] += 100/maxrounds
                                    pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                    key_idx=0
                                    dfc2[m2,i,j]=spearmanr(vals2[i,starttimes2[m2]:endtimes2[m2]],vals2[j,starttimes2[m2]+delayval:endtimes2[m2]+delayval]).correlation                                    
                elif raw_or_epoch.get()==3:
                    event_names=list(event_dict.keys())
                    n_chans=len(x1.ch_names)
                    delayval=int(float(delay.get())*x1.info['sfreq']/1000)
                    wlenval=int(float(wlen.get())*x1.info['sfreq']/1000)
                    woverlapval=int(float(woverlap.get())*x1.info['sfreq']/1000)
                    vals1=[]
                    n_epochs1=[]
                    for i in range(len(event_names)):
                        vals1.append(x1[event_names[i]].get_data())
                        n_epochs1.append(vals1[i].shape[0])
                    starttimes1=np.arange(0,vals1[0].shape[2]-wlenval-delayval,wlenval-woverlapval)
                    endtimes1=starttimes1+wlenval
                    tdfc1=[]
                    for i in range(len(event_names)):
                        tdfc1.append(np.empty((n_epochs1[i],len(starttimes1),n_chans,n_chans)))
                        tdfc1[i].fill(np.nan)
                    dfc1=np.empty((len(event_names),len(starttimes1),n_chans,n_chans))
                    dfc1.fill(np.nan)
                    if x2 is not None:
                        vals2=[]
                        n_epochs2=[]
                        for i in range(len(event_names)):
                            vals2.append(x2[event_names[i]].get_data())
                            n_epochs2.append(vals2[i].shape[0])
                        starttimes2=np.arange(0,vals2[0].shape[2]-wlenval-delayval,wlenval-woverlapval)
                        endtimes2=starttimes2+wlenval
                        tdfc2=[]
                        for i in range(len(event_names)):
                            tdfc2.append(np.empty((n_epochs2[i],len(starttimes1),n_chans,n_chans)))
                            tdfc2[i].fill(np.nan)
                        dfc2=np.empty((len(event_names),len(starttimes2),n_chans,n_chans))
                        dfc2.fill(np.nan)
                    pbar['value']=0.0
                    maxrounds=0
                    for ev in range(len(event_names)):
                        for i in range(n_chans):
                            for j in range(n_chans):
                                for m1 in range(len(starttimes1)):
                                    for e1 in range(n_epochs1[ev]):
                                        maxrounds+=1
                                if x2 is not None:
                                    for m2 in range(len(starttimes2)):
                                        for e2 in range(n_epochs2[ev]):
                                            maxrounds+=1
                    k=0                
                    for ev in range(len(event_names)):
                        for i in range(n_chans):
                            for j in range(n_chans):
                                for m1 in range(len(starttimes1)):
                                    for e1 in range(n_epochs1[ev]):
                                        win.update_idletasks()
                                        win.update()
                                        k+=1
                                        pbar['value'] += 100/maxrounds
                                        pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                        key_idx=0
                                        tdfc1[ev][e1,m1,i,j]=spearmanr(vals1[ev][e1,i,int(starttimes1[m1]):int(endtimes1[m1])],vals1[ev][e1,j,int(starttimes1[m1])+delayval:int(endtimes1[m1])+delayval]).correlation
                                if x2 is not None:
                                    for m2 in range(len(starttimes2)):
                                        for e2 in range(n_epochs2[ev]):
                                            win.update_idletasks()
                                            win.update()
                                            k+=1
                                            pbar['value'] += 100/maxrounds
                                            pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                            key_idx=0
                                            tdfc2[ev][e2,m2,i,j]=spearmanr(vals2[ev][e2,i,starttimes2[m2]:endtimes2[m2]],vals2[ev][e2,j,starttimes2[m2]+delayval:endtimes2[m2]+delayval]).correlation
                    pbtxt['text']="finishing, please wait"
                    for ev in range(len(event_names)):
                        for i in range(n_chans):
                            for j in range(n_chans):
                                for m1 in range(len(starttimes1)):
                                    dfc1[ev,m1,i,j]=np.nanmean(tdfc1[ev][:,m1,i,j])
                                if x2 is not None:
                                    for m2 in range(len(starttimes2)):
                                        dfc2[ev,m2,i,j]=np.nanmean(tdfc2[ev][:,m2,i,j])        
                    pbtxt['text']="done!"
                if error2==0:
                    def make_film():
                        winfilm=Toplevel(win)
                        Label(winfilm,text="Make Dynamic Functional Connectivity animation").grid(row=0,column=0,padx=10,pady=10,columnspan=3)
                        pbar2=Progressbar(winfilm,orient=HORIZONTAL,length=100,mode='determinate')
                        pbar2['value']=0.0
                        pbtxt2=Label(winfilm,text="--")
                        def process_frames():
                            if raw_or_epoch.get()==3:
                                nocond2=0
                                try:
                                    n_frames=min(len(dfc1[0,:,0,0]),len(dfc2[0,:,0,0]))
                                except:
                                    n_frames=len(dfc1[0,:,0,0])
                                    nocond2=1
                                k=0
                                os.mkdir('temp_DFC_frames')
                                for frame in range(n_frames):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar2['value'] += 100/n_frames
                                    pbtxt2['text']=f"{k:d}/{n_frames:d}"
                                    y1=[]
                                    if nocond2==1:
                                        y2=None
                                    else:
                                        y2=[]
                                    for ev in range(len(event_names)):
                                        y1.append(pd.DataFrame(dfc1[ev,frame,:,:],index=x1.ch_names,columns=x1.ch_names))
                                        if nocond2==0:
                                            y2.append(pd.DataFrame(dfc2[ev,frame,:,:],index=x2.ch_names,columns=x2.ch_names))
                                    make_frame(y1,y2,vmin=-1,vmax=1,color='bwr',lblttl='Spearman correlation',frttl=f'temp_DFC_frames/DFC_frame{frame:05d}.png')
                                pbtxt2['text']="Finishing up"
                                command=f'ffmpeg -framerate {fr.get()} -i temp_DFC_frames/DFC_frame%05d.png -pix_fmt yuv420p DFC_Spearman.mp4'
                                os.system(command)
                                shutil.rmtree('temp_DFC_frames')
                                showinfo(title="Info",message="Video of DFC was saved as DFC_Spearman.mp4 at\n"+os.getcwd())
                            else:
                                nocond2=0
                                n_frames1=len(dfc1[:,0,0])
                                total_frames=n_frames1
                                if raw_or_epoch.get()==2:
                                    ch_names=x1.ch_names
                                else:
                                    ch_names=xraw1.ch_names
                                try:
                                    n_frames2=len(dfc2[:,0,0])
                                    total_frames+=n_frames2
                                except:
                                    nocond2=1
                                k=0
                                os.mkdir('temp_DFC_frames')
                                for frame in range(n_frames1):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar2['value'] += 100/total_frames
                                    pbtxt2['text']=f"{k:d}/{total_frames:d}"
                                    y1=pd.DataFrame(dfc1[frame,:,:],index=ch_names,columns=ch_names)
                                    make_frame2(y1,vmin=-1,vmax=1,color='bwr',lblttl='Spearman correlation',frttl=f'temp_DFC_frames/DFC_{cond1_name.get()}_frame{frame:05d}.png')
                                if nocond==0:
                                    for frame in range(n_frames2):
                                        win.update_idletasks()
                                        win.update()
                                        k+=1
                                        pbar2['value'] += 100/total_frames
                                        pbtxt2['text']=f"{k:d}/{total_frames:d}"
                                        y2=pd.DataFrame(dfc2[frame,:,:],index=ch_names,columns=ch_names)
                                        make_frame2(y2,vmin=-1,vmax=1,color='bwr',lblttl='Spearman correlation',frttl=f'temp_DFC_frames/DFC_{cond2_name.get()}_frame{frame:05d}.png')
                                pbtxt2['text']="Finishing up"
                                command=f'ffmpeg -framerate {fr.get()} -i temp_DFC_frames/DFC_{cond1_name.get()}_frame%05d.png -pix_fmt yuv420p DFC_{cond1_name.get()}_Spearman.mp4'
                                os.system(command)
                                if nocond==0:
                                    command=f'ffmpeg -framerate {fr.get()} -i temp_DFC_frames/DFC_{cond2_name.get()}_frame%05d.png -pix_fmt yuv420p DFC_{cond2_name.get()}_Spearman.mp4'
                                    os.system(command)
                                shutil.rmtree('temp_DFC_frames')
                                if nocond==0:
                                    showinfo(title="Info",message="Videos of DFC were saved as\nDFC_{cond1_name.get()}_Spearman.mp4 and DFC_{cond1_name.get()}_Spearman.mp4 at\n"+os.getcwd())
                                else:
                                    showinfo(title="Info",message="Video of DFC were saved as\nDFC_{cond1_name.get()}_Spearman.mp4 at\n"+os.getcwd())
                            winfilm.destroy()
                        btn2=Button(winfilm, text='Create video', command=process_frames)
                        btn2.grid(row=1,column=0,sticky=W,padx=10)
                        pbar2.grid(row=1,column=1,sticky=W)
                        pbtxt2.grid(row=1,column=2,sticky=W)
                    def save_corr():
                        showinfo(title="Info",message="The full results will be saved as a\nNumpy array of size (n_times,n_chans,n_chans))")
                        fname1 = fd.asksaveasfilename(title="Spearman correlation DFC condition "+cond1_name.get(),defaultextension=".npy",filetypes=(("Numpy array", "*.npy"),("All Files", "*.*")))
                        np.save(fname1,dfc1)
                        if (x2 is not None) or (raw2 is not None):
                            fname2 = fd.asksaveasfilename(title="Spearman correlation DFC condition "+cond2_name.get(),defaultextension=".npy",filetypes=(("Numpy array", "*.npy"),("All Files", "*.*")))
                            np.save(fname2,dfc2)
                    Button(win,text="Save results",command=save_corr).grid(row=10,column=0,padx=10,sticky=W)
                    Button(win,text="Make DFC animations",command=make_film).grid(row=11,column=0,padx=10,sticky=W)
                    Button(win,text="Close",command=win.destroy).grid(row=12,column=0,padx=10,pady=10,columnspan=5)
            else:
                showinfo(title="Error",message="Delay, window length and frame rate must be positive numbers")
        Label(win,text="Calculate correlations").grid(row=0,column=0,columnspan=4,padx=10,pady=10)
        Label(win,text="Calculate from:").grid(row=1,column=0,padx=10,sticky=W)
        Radiobutton(win,text="From raw EEG",variable=raw_or_epoch,value=1).grid(row=1,column=1,sticky=W)
        Radiobutton(win,text="From epoch:",variable=raw_or_epoch,value=2).grid(row=2,column=1,sticky=W)
        Entry(win,textvariable=sel_epoch1,width=2).grid(row=2,column=2,sticky=W)
        Label(win,text=f"for {cond1_name.get()}").grid(row=2,column=3,sticky=W)
        Entry(win,textvariable=sel_epoch2,width=2).grid(row=3,column=2,sticky=W)
        Label(win,text=f"for {cond2_name.get()}").grid(row=3,column=3,sticky=W)
        Radiobutton(win,text="Average of results of all epochs",variable=raw_or_epoch,value=3).grid(row=4,column=1,sticky=W,columnspan=3)
        Label(win,text="Transmission delay between brain regions (ms):").grid(row=5,column=0,padx=10,sticky=W)
        Entry(win,textvariable=delay,width=4).grid(row=5,column=1,sticky=W)
        Label(win,text="Moving window length (ms):").grid(row=6,column=0,padx=10,sticky=W)
        Entry(win,textvariable=wlen,width=6).grid(row=6,column=1,sticky=W)
        Label(win,text="Moving window overlap (ms):").grid(row=7,column=0,padx=10,sticky=W)
        Entry(win,textvariable=woverlap,width=6).grid(row=7,column=1,sticky=W)
        Label(win,text="Frame rate:").grid(row=8,column=0,padx=10,sticky=W)
        Entry(win,textvariable=fr,width=6).grid(row=8,column=1,sticky=W)
        pbar=Progressbar(win,orient=HORIZONTAL,length=100,mode='determinate')
        pbar['value']=0.0
        pbtxt=Label(win,text="--")
        btn=Button(win, text='Start calculation', command=step)
        btn.grid(row=9,column=0,sticky=W,padx=10)
        pbar.grid(row=9,column=1,sticky=W)
        pbtxt.grid(row=9,column=2,sticky=W,columnspan=3)

#transfer entropy based dynamic functional connectivity
def te_dfc():
    win=Toplevel(main)
    raw_or_epoch=IntVar()
    raw_or_epoch.set(3)
    delay=StringVar()
    delay.set("10")
    wlen=StringVar()
    wlen.set("500")
    woverlap=StringVar()
    woverlap.set("250")
    sel_epoch1=StringVar()
    sel_epoch1.set("1")
    sel_epoch2=StringVar()
    sel_epoch2.set("1")
    ns=StringVar()
    ns.set('5')
    div_type=IntVar()
    div_type.set(1)
    xdiv_vals=StringVar()
    xdiv_vals.set('-30,-10,10,30')
    ydiv_vals=StringVar()
    ydiv_vals.set('-30,-10,10,30')
    lxp=StringVar()
    lxp.set('1')
    lxp=StringVar()
    lxp.set('1')
    lyp=StringVar()
    lyp.set('1')
    lyf=StringVar()
    lyf.set('1')
    tau=StringVar()
    tau.set('1')
    unit=StringVar()
    optionlist=['bits','nat','ban']
    fr=StringVar()
    fr.set("5")
    error=make_x()
    if (error==1) and (raw1 is None):
        Label(win,text="ERROR\nIt is required at least\n1 preprocessed or raw data",justify=CENTER).grid(row=0,column=0,padx=10,pady=10)        
        Button(win,text="OK",command=win.destroy).grid(row=1,column=0,padx=10)
    else:
        def step():
            if (int(delay.get())>=0) and (int(wlen.get())>0) and (int(fr.get())>0):
                error2=0
                if div_type.get()==1:
                    symb_type='equal-divs'
                    x_divs,y_divs=None,None
                elif div_type.get()==2:
                    symb_type='equal-points'
                    x_divs,y_divs=None,None
                elif div_type.get()==3:
                    x_divs=[float(i) for i in xdiv_vals.get().split(sep=',')]
                    y_divs=[float(i) for i in ydiv_vals.get().split(sep=',')]
                    symb_type=None
                if (raw_or_epoch.get()==2) and (error==1):
                    showinfo(title="Error",message="To work with epochs it is\nnecessary to have at\nleast 1 preprocessed data")
                    error2=1
                elif (raw_or_epoch.get()==1) and (raw1 is None):
                    showinfo(title="Error",message="To work with raw it is\nnecessary to have loaded at\nleast 1 raw EDF file")
                    error2=1
                elif raw_or_epoch.get()==1:
                    n_chans=len(xraw1.ch_names)
                    delayval=int(float(delay.get())*xraw1.info['sfreq']/1000)
                    wlenval=int(float(wlen.get())*xraw1.info['sfreq']/1000)
                    woverlapval=int(float(woverlap.get())*xraw1.info['sfreq']/1000)
                    vals1=xraw1.get_data()
                    starttimes1=np.arange(0,vals1.shape[1]-wlenval-delayval,wlenval-woverlapval)
                    endtimes1=starttimes1+wlenval
                    dfc1=np.empty((len(starttimes1),n_chans,n_chans))
                    dfc1.fill(np.nan)
                    if xraw2 is not None:
                        vals2=xraw2.get_data()
                        starttimes2=np.arange(0,vals2.shape[1]-wlenval-delayval,wlenval-woverlapval)
                        endtimes2=starttimes2+wlenval
                        dfc2=np.empty((len(starttimes2),n_chans,n_chans))
                        dfc2.fill(np.nan)
                    pbar['value']=0.0
                    maxrounds=0
                    for i in range(n_chans):
                        for j in range(n_chans):
                            for m1 in range(len(starttimes1)):
                                maxrounds+=1
                            if xraw2 is not None:
                                for m2 in range(len(starttimes2)):
                                    maxrounds+=1
                    k=0                
                    for i in range(n_chans):
                        for j in range(n_chans):
                            for m1 in range(len(starttimes1)):
                                win.update_idletasks()
                                win.update()
                                k+=1
                                pbar['value'] += 100/maxrounds
                                pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                key_idx=0
                                dfc1[m1,i,j]=transfer_entropy(vals1[i,starttimes1[m1]:endtimes1[m1]],vals1[j,starttimes1[m1]+delayval:endtimes1[m1]+delayval],
                                                                                symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                symbolic_length=(int(lxp.get()),int(lyp.get()),int(lyf.get())),
                                                                                x_divs=x_divs,y_divs=y_divs,units=unit.get())
                            if xraw2 is not None:
                                for m2 in range(len(starttimes2)):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar['value'] += 100/maxrounds
                                    pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                    key_idx=0
                                    dfc2[m2,i,j]=transfer_entropy(vals2[i,starttimes2[m2]:endtimes2[m2]],vals2[j,starttimes2[m2]+delayval:endtimes2[m2]+delayval],
                                                                                symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                symbolic_length=(int(lxp.get()),int(lyp.get()),int(lyf.get())),
                                                                                x_divs=x_divs,y_divs=y_divs,units=unit.get())                                   
                elif raw_or_epoch.get()==2:
                    n_chans=len(x1.ch_names)
                    delayval=int(float(delay.get())*x1.info['sfreq']/1000)
                    wlenval=int(float(wlen.get())*x1.info['sfreq']/1000)
                    woverlapval=int(float(woverlap.get())*x1.info['sfreq']/1000)
                    vals1=x1.get_data()[int(sel_epoch1.get()),:,:]
                    starttimes1=np.arange(0,vals1.shape[1]-wlenval-delayval,wlenval-woverlapval)
                    endtimes1=starttimes1+wlenval
                    dfc1=np.empty((len(starttimes1),n_chans,n_chans))
                    dfc1.fill(np.nan)
                    if x2 is not None:
                        vals2=x2.get_data()[int(sel_epoch2.get()),:,:]
                        starttimes2=np.arange(0,vals2.shape[1]-wlenval-delayval,wlenval-woverlapval)
                        endtimes2=starttimes2+wlenval
                        dfc2=np.empty((len(starttimes2),n_chans,n_chans))
                        dfc2.fill(np.nan)
                    pbar['value']=0.0
                    maxrounds=0
                    for i in range(n_chans):
                        for j in range(n_chans):
                            for m1 in range(len(starttimes1)):
                                maxrounds+=1
                            if x2 is not None:
                                for m2 in range(len(starttimes2)):
                                    maxrounds+=1
                    k=0                
                    for i in range(n_chans):
                        for j in range(n_chans):
                            for m1 in range(len(starttimes1)):
                                win.update_idletasks()
                                win.update()
                                k+=1
                                pbar['value'] += 100/maxrounds
                                pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                key_idx=0
                                dfc1[m1,i,j]=transfer_entropy(vals1[i,starttimes1[m1]:endtimes1[m1]],vals1[j,starttimes1[m1]+delayval:endtimes1[m1]+delayval],
                                                                                symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                symbolic_length=(int(lxp.get()),int(lyp.get()),int(lyf.get())),
                                                                                x_divs=x_divs,y_divs=y_divs,units=unit.get())
                            if x2 is not None:
                                for m2 in range(len(starttimes2)):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar['value'] += 100/maxrounds
                                    pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                    key_idx=0
                                    dfc2[m2,i,j]=transfer_entropy(vals2[i,starttimes2[m2]:endtimes2[m2]],vals2[j,starttimes2[m2]+delayval:endtimes2[m2]+delayval],
                                                                                symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                symbolic_length=(int(lxp.get()),int(lyp.get()),int(lyf.get())),
                                                                                x_divs=x_divs,y_divs=y_divs,units=unit.get())                                    
                elif raw_or_epoch.get()==3:
                    event_names=list(event_dict.keys())
                    n_chans=len(x1.ch_names)
                    delayval=int(float(delay.get())*x1.info['sfreq']/1000)
                    wlenval=int(float(wlen.get())*x1.info['sfreq']/1000)
                    woverlapval=int(float(woverlap.get())*x1.info['sfreq']/1000)
                    vals1=[]
                    n_epochs1=[]
                    for i in range(len(event_names)):
                        vals1.append(x1[event_names[i]].get_data())
                        n_epochs1.append(vals1[i].shape[0])
                    starttimes1=np.arange(0,vals1[0].shape[2]-wlenval-delayval,wlenval-woverlapval)
                    endtimes1=starttimes1+wlenval
                    tdfc1=[]
                    for i in range(len(event_names)):
                        tdfc1.append(np.empty((n_epochs1[i],len(starttimes1),n_chans,n_chans)))
                        tdfc1[i].fill(np.nan)
                    dfc1=np.empty((len(event_names),len(starttimes1),n_chans,n_chans))
                    dfc1.fill(np.nan)
                    if x2 is not None:
                        vals2=[]
                        n_epochs2=[]
                        for i in range(len(event_names)):
                            vals2.append(x2[event_names[i]].get_data())
                            n_epochs2.append(vals2[i].shape[0])
                        starttimes2=np.arange(0,vals2[0].shape[2]-wlenval-delayval,wlenval-woverlapval)
                        endtimes2=starttimes2+wlenval
                        tdfc2=[]
                        for i in range(len(event_names)):
                            tdfc2.append(np.empty((n_epochs2[i],len(starttimes1),n_chans,n_chans)))
                            tdfc2[i].fill(np.nan)
                        dfc2=np.empty((len(event_names),len(starttimes2),n_chans,n_chans))
                        dfc2.fill(np.nan)
                    pbar['value']=0.0
                    maxrounds=0
                    for ev in range(len(event_names)):
                        for i in range(n_chans):
                            for j in range(n_chans):
                                for m1 in range(len(starttimes1)):
                                    for e1 in range(n_epochs1[ev]):
                                        maxrounds+=1
                                if x2 is not None:
                                    for m2 in range(len(starttimes2)):
                                        for e2 in range(n_epochs2[ev]):
                                            maxrounds+=1
                    k=0                
                    for ev in range(len(event_names)):
                        for i in range(n_chans):
                            for j in range(n_chans):
                                for m1 in range(len(starttimes1)):
                                    for e1 in range(n_epochs1[ev]):
                                        win.update_idletasks()
                                        win.update()
                                        k+=1
                                        pbar['value'] += 100/maxrounds
                                        pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                        key_idx=0
                                        tdfc1[ev][e1,m1,i,j]=transfer_entropy(vals1[ev][e1,i,int(starttimes1[m1]):int(endtimes1[m1])],vals1[ev][e1,j,int(starttimes1[m1])+delayval:int(endtimes1[m1])+delayval],
                                                                                symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                symbolic_length=(int(lxp.get()),int(lyp.get()),int(lyf.get())),
                                                                                x_divs=x_divs,y_divs=y_divs,units=unit.get())
                                if x2 is not None:
                                    for m2 in range(len(starttimes2)):
                                        for e2 in range(n_epochs2[ev]):
                                            win.update_idletasks()
                                            win.update()
                                            k+=1
                                            pbar['value'] += 100/maxrounds
                                            pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                            key_idx=0
                                            tdfc2[ev][e2,m2,i,j]=transfer_entropy(vals2[ev][e2,i,int(starttimes2[m2]):int(endtimes2[m2])],vals2[ev][e2,j,int(starttimes2[m2])+delayval:int(endtimes2[m2])+delayval],
                                                                                symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                symbolic_length=(int(lxp.get()),int(lyp.get()),int(lyf.get())),
                                                                                x_divs=x_divs,y_divs=y_divs,units=unit.get())
                    pbtxt['text']="finishing, please wait"
                    for ev in range(len(event_names)):
                        for i in range(n_chans):
                            for j in range(n_chans):
                                for m1 in range(len(starttimes1)):
                                    dfc1[ev,m1,i,j]=np.nanmean(tdfc1[ev][:,m1,i,j])
                                if x2 is not None:
                                    for m2 in range(len(starttimes2)):
                                        dfc2[ev,m2,i,j]=np.nanmean(tdfc2[ev][:,m2,i,j])        
                    pbtxt['text']="done!"
                if error2==0:
                    def make_film():
                        winfilm=Toplevel(win)
                        Label(winfilm,text="Make Dynamic Functional Connectivity animation").grid(row=0,column=0,padx=10,pady=10,columnspan=3)
                        pbar2=Progressbar(winfilm,orient=HORIZONTAL,length=100,mode='determinate')
                        pbar2['value']=0.0
                        pbtxt2=Label(winfilm,text="--")
                        def process_frames():
                            if raw_or_epoch.get()==3:
                                nocond2=0
                                try:
                                    n_frames=min(len(dfc1[0,:,0,0]),len(dfc2[0,:,0,0]))
                                except:
                                    n_frames=len(dfc1[0,:,0,0])
                                    nocond2=1
                                k=0
                                os.mkdir('temp_DFC_frames')
                                for frame in range(n_frames):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar2['value'] += 100/n_frames
                                    pbtxt2['text']=f"{k:d}/{n_frames:d}"
                                    y1=[]
                                    if nocond2==1:
                                        y2=None
                                    else:
                                        y2=[]
                                    maxte=0
                                    for ev in range(len(event_names)):
                                        y1.append(pd.DataFrame(dfc1[ev,frame,:,:],index=x1.ch_names,columns=x1.ch_names))
                                        maxte=max(maxte,y1[ev].max().max())
                                        if nocond2==0:
                                            y2.append(pd.DataFrame(dfc2[ev,frame,:,:],index=x2.ch_names,columns=x2.ch_names))
                                            maxte=max(maxte,y2[ev].max().max())
                                    make_frame(y1,y2,vmin=0,vmax=maxte,color='Reds',lblttl='Transfer entropy',frttl=f'temp_DFC_frames/DFC_frame{frame:05d}.png')
                                pbtxt2['text']="Finishing up"
                                command=f'ffmpeg -framerate {fr.get()} -i temp_DFC_frames/DFC_frame%05d.png -pix_fmt yuv420p DFC_Transfer_Entropy.mp4'
                                os.system(command)
                                shutil.rmtree('temp_DFC_frames')
                                showinfo(title="Info",message="Video of DFC was saved as DFC_Transfer_Entropy.mp4 at\n"+os.getcwd())
                            else:
                                nocond2=0
                                n_frames1=len(dfc1[:,0,0])
                                total_frames=n_frames1
                                if raw_or_epoch.get()==2:
                                    ch_names=x1.ch_names
                                else:
                                    ch_names=xraw1.ch_names
                                try:
                                    n_frames2=len(dfc2[:,0,0])
                                    total_frames+=n_frames2
                                except:
                                    nocond2=1
                                k=0
                                os.mkdir('temp_DFC_frames')
                                for frame in range(n_frames1):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar2['value'] += 100/total_frames
                                    pbtxt2['text']=f"{k:d}/{total_frames:d}"
                                    y1=pd.DataFrame(dfc1[frame,:,:],index=ch_names,columns=ch_names)
                                    make_frame2(y1,vmin=0,vmax=y1.max().max(),color='Reds',lblttl='Transfer Entropy',frttl=f'temp_DFC_frames/DFC_{cond1_name.get()}_frame{frame:05d}.png')
                                if nocond==0:
                                    for frame in range(n_frames2):
                                        win.update_idletasks()
                                        win.update()
                                        k+=1
                                        pbar2['value'] += 100/total_frames
                                        pbtxt2['text']=f"{k:d}/{total_frames:d}"
                                        y2=pd.DataFrame(dfc2[frame,:,:],index=ch_names,columns=ch_names)
                                        make_frame2(y2,vmin=0,vmax=y2.max().max(),color='Reds',lblttl='Transfer Entropy',frttl=f'temp_DFC_frames/DFC_{cond2_name.get()}_frame{frame:05d}.png')
                                pbtxt2['text']="Finishing up"
                                command=f'ffmpeg -framerate {fr.get()} -i temp_DFC_frames/DFC_{cond1_name.get()}_frame%05d.png -pix_fmt yuv420p DFC_{cond1_name.get()}_Transfer_Entropy.mp4'
                                os.system(command)
                                if nocond==0:
                                    command=f'ffmpeg -framerate {fr.get()} -i temp_DFC_frames/DFC_{cond2_name.get()}_frame%05d.png -pix_fmt yuv420p DFC_{cond2_name.get()}_Transfer_Entropy.mp4'
                                    os.system(command)
                                shutil.rmtree('temp_DFC_frames')
                                if nocond==0:
                                    showinfo(title="Info",message="Videos of DFC were saved as\nDFC_{cond1_name.get()}_Transfer_Entropy.mp4 and DFC_{cond1_name.get()}_Transfer_Entropy.mp4 at\n"+os.getcwd())
                                else:
                                    showinfo(title="Info",message="Video of DFC were saved as\nDFC_{cond1_name.get()}_Transfer_Entropy.mp4 at\n"+os.getcwd())
                            winfilm.destroy()
                        btn2=Button(winfilm, text='Create video', command=process_frames)
                        btn2.grid(row=1,column=0,sticky=W,padx=10)
                        pbar2.grid(row=1,column=1,sticky=W)
                        pbtxt2.grid(row=1,column=2,sticky=W)
                    def save_corr():
                        showinfo(title="Info",message="The full results will be saved as a\nNumpy array of size (n_times,n_chans,n_chans))")
                        fname1 = fd.asksaveasfilename(title="Transfer Entropy DFC condition "+cond1_name.get(),defaultextension=".npy",filetypes=(("Numpy array", "*.npy"),("All Files", "*.*")))
                        np.save(fname1,dfc1)
                        if (x2 is not None) or (raw2 is not None):
                            fname2 = fd.asksaveasfilename(title="Transfer Entropy DFC condition "+cond2_name.get(),defaultextension=".npy",filetypes=(("Numpy array", "*.npy"),("All Files", "*.*")))
                            np.save(fname2,dfc2)
                    Button(win,text="Save results",command=save_corr).grid(row=20,column=0,padx=10,sticky=W)
                    Button(win,text="Make DFC animations",command=make_film).grid(row=21,column=0,padx=10,sticky=W)
                    Button(win,text="Close",command=win.destroy).grid(row=22,column=0,padx=10,pady=10,columnspan=5)
            else:
                showinfo(title="Error",message="Delay, window length and frame rate must be positive numbers")
        Label(win,text="Calculate transfer entropy").grid(row=0,column=0,columnspan=4,padx=10,pady=10)
        Label(win,text="Calculate from:").grid(row=1,column=0,padx=10,sticky=W)
        Radiobutton(win,text="From raw EEG",variable=raw_or_epoch,value=1).grid(row=1,column=1,sticky=W)
        Radiobutton(win,text="From epoch:",variable=raw_or_epoch,value=2).grid(row=2,column=1,sticky=W)
        Entry(win,textvariable=sel_epoch1,width=2).grid(row=2,column=2,sticky=W)
        Label(win,text=f"for {cond1_name.get()}").grid(row=2,column=3,sticky=W)
        Entry(win,textvariable=sel_epoch2,width=2).grid(row=3,column=2,sticky=W)
        Label(win,text=f"for {cond2_name.get()}").grid(row=3,column=3,sticky=W)
        Radiobutton(win,text="Average of results of all epochs",variable=raw_or_epoch,value=3).grid(row=4,column=1,sticky=W,columnspan=3)
        Label(win,text="Transmission delay between brain regions (ms):").grid(row=5,column=0,padx=10,sticky=W)
        Entry(win,textvariable=delay,width=4).grid(row=5,column=1,sticky=W)
        Label(win,text="Number of symbols (partition divisions):").grid(row=6,column=0,sticky=W,padx=10)
        Entry(win,textvariable=ns,width=3).grid(row=6,column=1,sticky=W)
        Label(win,text="Partition divisions:").grid(row=7,column=0,sticky=W,padx=10)
        Radiobutton(win,text="equal-sized divisions",variable=div_type,value=1).grid(row=7,column=1,sticky=W)
        Radiobutton(win,text="divisions with same number of points",variable=div_type,value=2).grid(row=8,column=1,columnspan=4,sticky=W)
        Radiobutton(win,text="other (separate with commas):",variable=div_type,value=3).grid(row=9,column=1,sticky=W)
        Label(win,text=" X divisions:").grid(row=9,column=2,sticky=W)
        Entry(win,textvariable=xdiv_vals,width=20).grid(row=9,column=3,sticky=W)
        Label(win,text=" Y divisions:").grid(row=10,column=2,sticky=W)
        Entry(win,textvariable=ydiv_vals,width=20).grid(row=10,column=3,sticky=W)
        Label(win,text="Symbolic length:").grid(row=11,column=0,sticky=W,padx=10)
        Label(win,text="Past of X:").grid(row=11,column=1,sticky=W)
        Entry(win,textvariable=lxp,width=3).grid(row=11,column=2,sticky=W)
        Label(win,text="Past of Y:").grid(row=12,column=1,sticky=W)
        Entry(win,textvariable=lyp,width=3).grid(row=12,column=2,sticky=W)
        Label(win,text="Future of Y:").grid(row=13,column=1,sticky=W)
        Entry(win,textvariable=lyf,width=3).grid(row=13,column=2,sticky=W)
        Label(win,text="Tau:").grid(row=14,column=0,sticky=W,padx=10)
        Entry(win,textvariable=tau,width=3).grid(row=14,column=1,sticky=W)
        Label(win,text="Units:").grid(row=15,column=0,sticky=W,padx=10)
        OptionMenu(win,unit,*optionlist).grid(row=15,column=1,sticky=W)
        Label(win,text="Moving window length (ms):").grid(row=16,column=0,padx=10,sticky=W)
        Entry(win,textvariable=wlen,width=6).grid(row=16,column=1,sticky=W)
        Label(win,text="Moving window overlap (ms):").grid(row=17,column=0,padx=10,sticky=W)
        Entry(win,textvariable=woverlap,width=6).grid(row=17,column=1,sticky=W)
        Label(win,text="Frame rate:").grid(row=18,column=0,padx=10,sticky=W)
        Entry(win,textvariable=fr,width=6).grid(row=18,column=1,sticky=W)
        pbar=Progressbar(win,orient=HORIZONTAL,length=100,mode='determinate')
        pbar['value']=0.0
        pbtxt=Label(win,text="--")
        btn=Button(win, text='Start calculation', command=step)
        btn.grid(row=19,column=0,sticky=W,padx=10)
        pbar.grid(row=19,column=1,sticky=W)
        pbtxt.grid(row=19,column=2,sticky=W,columnspan=3)

#mutual information based dynamic functional connectivity
def mi_dfc():
    win=Toplevel(main)
    raw_or_epoch=IntVar()
    raw_or_epoch.set(3)
    delay=StringVar()
    delay.set("10")
    wlen=StringVar()
    wlen.set("500")
    woverlap=StringVar()
    woverlap.set("250")
    sel_epoch1=StringVar()
    sel_epoch1.set("1")
    sel_epoch2=StringVar()
    sel_epoch2.set("1")
    ns=StringVar()
    ns.set('5')
    div_type=IntVar()
    div_type.set(1)
    xdiv_vals=StringVar()
    xdiv_vals.set('-30,-10,10,30')
    ydiv_vals=StringVar()
    ydiv_vals.set('-30,-10,10,30')
    lxp=StringVar()
    lxp.set('1')
    lxp=StringVar()
    lxp.set('1')
    lyp=StringVar()
    lyp.set('1')
    tau=StringVar()
    tau.set('1')
    unit=StringVar()
    optionlist=['bits','nat','ban']
    fr=StringVar()
    fr.set("5")
    error=make_x()
    if (error==1) and (raw1 is None):
        Label(win,text="ERROR\nIt is required at least\n1 preprocessed or raw data",justify=CENTER).grid(row=0,column=0,padx=10,pady=10)        
        Button(win,text="OK",command=win.destroy).grid(row=1,column=0,padx=10)
    else:
        def step():
            if (int(delay.get())>=0) and (int(wlen.get())>0) and (int(fr.get())>0):
                error2=0
                if div_type.get()==1:
                    symb_type='equal-divs'
                    x_divs,y_divs=None,None
                elif div_type.get()==2:
                    symb_type='equal-points'
                    x_divs,y_divs=None,None
                elif div_type.get()==3:
                    x_divs=[float(i) for i in xdiv_vals.get().split(sep=',')]
                    y_divs=[float(i) for i in ydiv_vals.get().split(sep=',')]
                    symb_type=None
                if (raw_or_epoch.get()==2) and (error==1):
                    showinfo(title="Error",message="To work with epochs it is\nnecessary to have at\nleast 1 preprocessed data")
                    error2=1
                elif (raw_or_epoch.get()==1) and (raw1 is None):
                    showinfo(title="Error",message="To work with raw it is\nnecessary to have loaded at\nleast 1 raw EDF file")
                    error2=1
                elif raw_or_epoch.get()==1:
                    n_chans=len(xraw1.ch_names)
                    delayval=int(float(delay.get())*xraw1.info['sfreq']/1000)
                    wlenval=int(float(wlen.get())*xraw1.info['sfreq']/1000)
                    woverlapval=int(float(woverlap.get())*xraw1.info['sfreq']/1000)
                    vals1=xraw1.get_data()
                    starttimes1=np.arange(0,vals1.shape[1]-wlenval-delayval,wlenval-woverlapval)
                    endtimes1=starttimes1+wlenval
                    dfc1=np.empty((len(starttimes1),n_chans,n_chans))
                    dfc1.fill(np.nan)
                    if xraw2 is not None:
                        vals2=xraw2.get_data()
                        starttimes2=np.arange(0,vals2.shape[1]-wlenval-delayval,wlenval-woverlapval)
                        endtimes2=starttimes2+wlenval
                        dfc2=np.empty((len(starttimes2),n_chans,n_chans))
                        dfc2.fill(np.nan)
                    pbar['value']=0.0
                    maxrounds=0
                    for i in range(n_chans):
                        for j in range(n_chans):
                            for m1 in range(len(starttimes1)):
                                maxrounds+=1
                            if xraw2 is not None:
                                for m2 in range(len(starttimes2)):
                                    maxrounds+=1
                    k=0                
                    for i in range(n_chans):
                        for j in range(n_chans):
                            for m1 in range(len(starttimes1)):
                                win.update_idletasks()
                                win.update()
                                k+=1
                                pbar['value'] += 100/maxrounds
                                pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                key_idx=0
                                dfc1[m1,i,j]=mutual_info(vals1[i,starttimes1[m1]:endtimes1[m1]],vals1[j,starttimes1[m1]+delayval:endtimes1[m1]+delayval],
                                                                                symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                symbolic_length=(int(lxp.get()),int(lyp.get())),
                                                                                x_divs=x_divs,y_divs=y_divs,units=unit.get())
                            if xraw2 is not None:
                                for m2 in range(len(starttimes2)):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar['value'] += 100/maxrounds
                                    pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                    key_idx=0
                                    dfc2[m2,i,j]=mutual_info(vals2[i,starttimes2[m2]:endtimes2[m2]],vals2[j,starttimes2[m2]+delayval:endtimes2[m2]+delayval],
                                                                                symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                symbolic_length=(int(lxp.get()),int(lyp.get())),
                                                                                x_divs=x_divs,y_divs=y_divs,units=unit.get())                                   
                elif raw_or_epoch.get()==2:
                    n_chans=len(x1.ch_names)
                    delayval=int(float(delay.get())*x1.info['sfreq']/1000)
                    wlenval=int(float(wlen.get())*x1.info['sfreq']/1000)
                    woverlapval=int(float(woverlap.get())*x1.info['sfreq']/1000)
                    vals1=x1.get_data()[int(sel_epoch1.get()),:,:]
                    starttimes1=np.arange(0,vals1.shape[1]-wlenval-delayval,wlenval-woverlapval)
                    endtimes1=starttimes1+wlenval
                    dfc1=np.empty((len(starttimes1),n_chans,n_chans))
                    dfc1.fill(np.nan)
                    if x2 is not None:
                        vals2=x2.get_data()[int(sel_epoch2.get()),:,:]
                        starttimes2=np.arange(0,vals2.shape[1]-wlenval-delayval,wlenval-woverlapval)
                        endtimes2=starttimes2+wlenval
                        dfc2=np.empty((len(starttimes2),n_chans,n_chans))
                        dfc2.fill(np.nan)
                    pbar['value']=0.0
                    maxrounds=0
                    for i in range(n_chans):
                        for j in range(n_chans):
                            for m1 in range(len(starttimes1)):
                                maxrounds+=1
                            if x2 is not None:
                                for m2 in range(len(starttimes2)):
                                    maxrounds+=1
                    k=0                
                    for i in range(n_chans):
                        for j in range(n_chans):
                            for m1 in range(len(starttimes1)):
                                win.update_idletasks()
                                win.update()
                                k+=1
                                pbar['value'] += 100/maxrounds
                                pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                key_idx=0
                                dfc1[m1,i,j]=mutual_info(vals1[i,starttimes1[m1]:endtimes1[m1]],vals1[j,starttimes1[m1]+delayval:endtimes1[m1]+delayval],
                                                                                symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                symbolic_length=(int(lxp.get()),int(lyp.get())),
                                                                                x_divs=x_divs,y_divs=y_divs,units=unit.get())
                            if x2 is not None:
                                for m2 in range(len(starttimes2)):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar['value'] += 100/maxrounds
                                    pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                    key_idx=0
                                    dfc2[m2,i,j]=mutual_info(vals2[i,starttimes2[m2]:endtimes2[m2]],vals2[j,starttimes2[m2]+delayval:endtimes2[m2]+delayval],
                                                                                symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                symbolic_length=(int(lxp.get()),int(lyp.get())),
                                                                                x_divs=x_divs,y_divs=y_divs,units=unit.get())                                    
                elif raw_or_epoch.get()==3:
                    event_names=list(event_dict.keys())
                    n_chans=len(x1.ch_names)
                    delayval=int(float(delay.get())*x1.info['sfreq']/1000)
                    wlenval=int(float(wlen.get())*x1.info['sfreq']/1000)
                    woverlapval=int(float(woverlap.get())*x1.info['sfreq']/1000)
                    vals1=[]
                    n_epochs1=[]
                    for i in range(len(event_names)):
                        vals1.append(x1[event_names[i]].get_data())
                        n_epochs1.append(vals1[i].shape[0])
                    starttimes1=np.arange(0,vals1[0].shape[2]-wlenval-delayval,wlenval-woverlapval)
                    endtimes1=starttimes1+wlenval
                    tdfc1=[]
                    for i in range(len(event_names)):
                        tdfc1.append(np.empty((n_epochs1[i],len(starttimes1),n_chans,n_chans)))
                        tdfc1[i].fill(np.nan)
                    dfc1=np.empty((len(event_names),len(starttimes1),n_chans,n_chans))
                    dfc1.fill(np.nan)
                    if x2 is not None:
                        vals2=[]
                        n_epochs2=[]
                        for i in range(len(event_names)):
                            vals2.append(x2[event_names[i]].get_data())
                            n_epochs2.append(vals2[i].shape[0])
                        starttimes2=np.arange(0,vals2[0].shape[2]-wlenval-delayval,wlenval-woverlapval)
                        endtimes2=starttimes2+wlenval
                        tdfc2=[]
                        for i in range(len(event_names)):
                            tdfc2.append(np.empty((n_epochs2[i],len(starttimes1),n_chans,n_chans)))
                            tdfc2[i].fill(np.nan)
                        dfc2=np.empty((len(event_names),len(starttimes2),n_chans,n_chans))
                        dfc2.fill(np.nan)
                    pbar['value']=0.0
                    maxrounds=0
                    for ev in range(len(event_names)):
                        for i in range(n_chans):
                            for j in range(n_chans):
                                for m1 in range(len(starttimes1)):
                                    for e1 in range(n_epochs1[ev]):
                                        maxrounds+=1
                                if x2 is not None:
                                    for m2 in range(len(starttimes2)):
                                        for e2 in range(n_epochs2[ev]):
                                            maxrounds+=1
                    k=0                
                    for ev in range(len(event_names)):
                        for i in range(n_chans):
                            for j in range(n_chans):
                                for m1 in range(len(starttimes1)):
                                    for e1 in range(n_epochs1[ev]):
                                        win.update_idletasks()
                                        win.update()
                                        k+=1
                                        pbar['value'] += 100/maxrounds
                                        pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                        key_idx=0
                                        tdfc1[ev][e1,m1,i,j]=mutual_info(vals1[ev][e1,i,int(starttimes1[m1]):int(endtimes1[m1])],vals1[ev][e1,j,int(starttimes1[m1])+delayval:int(endtimes1[m1])+delayval],
                                                                                symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                symbolic_length=(int(lxp.get()),int(lyp.get())),
                                                                                x_divs=x_divs,y_divs=y_divs,units=unit.get())
                                if x2 is not None:
                                    for m2 in range(len(starttimes2)):
                                        for e2 in range(n_epochs2[ev]):
                                            win.update_idletasks()
                                            win.update()
                                            k+=1
                                            pbar['value'] += 100/maxrounds
                                            pbtxt['text']=f"{k:d}/{maxrounds:d}"
                                            key_idx=0
                                            tdfc2[ev][e2,m2,i,j]=mutual_info(vals2[ev][e2,i,int(starttimes2[m2]):int(endtimes2[m2])],vals2[ev][e2,j,int(starttimes2[m2])+delayval:int(endtimes2[m2])+delayval],
                                                                                symbolic_type=symb_type,n_symbols=int(ns.get()),tau=int(tau.get()),
                                                                                symbolic_length=(int(lxp.get()),int(lyp.get())),
                                                                                x_divs=x_divs,y_divs=y_divs,units=unit.get())
                    pbtxt['text']="finishing, please wait"
                    for ev in range(len(event_names)):
                        for i in range(n_chans):
                            for j in range(n_chans):
                                for m1 in range(len(starttimes1)):
                                    dfc1[ev,m1,i,j]=np.nanmean(tdfc1[ev][:,m1,i,j])
                                if x2 is not None:
                                    for m2 in range(len(starttimes2)):
                                        dfc2[ev,m2,i,j]=np.nanmean(tdfc2[ev][:,m2,i,j])        
                    pbtxt['text']="done!"
                if error2==0:
                    def make_film():
                        winfilm=Toplevel(win)
                        Label(winfilm,text="Make Dynamic Functional Connectivity animation").grid(row=0,column=0,padx=10,pady=10,columnspan=3)
                        pbar2=Progressbar(winfilm,orient=HORIZONTAL,length=100,mode='determinate')
                        pbar2['value']=0.0
                        pbtxt2=Label(winfilm,text="--")
                        def process_frames():
                            if raw_or_epoch.get()==3:
                                nocond2=0
                                try:
                                    n_frames=min(len(dfc1[0,:,0,0]),len(dfc2[0,:,0,0]))
                                except:
                                    n_frames=len(dfc1[0,:,0,0])
                                    nocond2=1
                                k=0
                                os.mkdir('temp_DFC_frames')
                                for frame in range(n_frames):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar2['value'] += 100/n_frames
                                    pbtxt2['text']=f"{k:d}/{n_frames:d}"
                                    y1=[]
                                    if nocond2==1:
                                        y2=None
                                    else:
                                        y2=[]
                                    maxmi=0
                                    for ev in range(len(event_names)):
                                        y1.append(pd.DataFrame(dfc1[ev,frame,:,:],index=x1.ch_names,columns=x1.ch_names))
                                        maxmi=max(maxmi,y1[ev].max().max())
                                        if nocond2==0:
                                            y2.append(pd.DataFrame(dfc2[ev,frame,:,:],index=x2.ch_names,columns=x2.ch_names))
                                            maxmi=max(maxmi,y2[ev].max().max())
                                    make_frame(y1,y2,vmin=0,vmax=maxmi,color='Reds',lblttl='Mutual Information',frttl=f'temp_DFC_frames/DFC_frame{frame:05d}.png')
                                pbtxt2['text']="Finishing up"
                                command=f'ffmpeg -framerate {fr.get()} -i temp_DFC_frames/DFC_frame%05d.png -pix_fmt yuv420p DFC_Mutual_Information.mp4'
                                os.system(command)
                                shutil.rmtree('temp_DFC_frames')
                                showinfo(title="Info",message="Video of DFC was saved as DFC_Mutual_Information.mp4 at\n"+os.getcwd())
                            else:
                                nocond2=0
                                n_frames1=len(dfc1[:,0,0])
                                total_frames=n_frames1
                                if raw_or_epoch.get()==2:
                                    ch_names=x1.ch_names
                                else:
                                    ch_names=xraw1.ch_names
                                try:
                                    n_frames2=len(dfc2[:,0,0])
                                    total_frames+=n_frames2
                                except:
                                    nocond2=1
                                k=0
                                os.mkdir('temp_DFC_frames')
                                for frame in range(n_frames1):
                                    win.update_idletasks()
                                    win.update()
                                    k+=1
                                    pbar2['value'] += 100/total_frames
                                    pbtxt2['text']=f"{k:d}/{total_frames:d}"
                                    y1=pd.DataFrame(dfc1[frame,:,:],index=ch_names,columns=ch_names)
                                    make_frame2(y1,vmin=0,vmax=y1.max().max(),color='Reds',lblttl='Mutual Information',frttl=f'temp_DFC_frames/DFC_{cond1_name.get()}_frame{frame:05d}.png')
                                if nocond==0:
                                    for frame in range(n_frames2):
                                        win.update_idletasks()
                                        win.update()
                                        k+=1
                                        pbar2['value'] += 100/total_frames
                                        pbtxt2['text']=f"{k:d}/{total_frames:d}"
                                        y2=pd.DataFrame(dfc2[frame,:,:],index=ch_names,columns=ch_names)
                                        make_frame2(y2,vmin=0,vmax=y2.max().max(),color='Reds',lblttl='Mutual Information',frttl=f'temp_DFC_frames/DFC_{cond2_name.get()}_frame{frame:05d}.png')
                                pbtxt2['text']="Finishing up"
                                command=f'ffmpeg -framerate {fr.get()} -i temp_DFC_frames/DFC_{cond1_name.get()}_frame%05d.png -pix_fmt yuv420p DFC_{cond1_name.get()}_Mutual_Information.mp4'
                                os.system(command)
                                if nocond==0:
                                    command=f'ffmpeg -framerate {fr.get()} -i temp_DFC_frames/DFC_{cond2_name.get()}_frame%05d.png -pix_fmt yuv420p DFC_{cond2_name.get()}_Mutual_Information.mp4'
                                    os.system(command)
                                shutil.rmtree('temp_DFC_frames')
                                if nocond==0:
                                    showinfo(title="Info",message="Videos of DFC were saved as\nDFC_{cond1_name.get()}_Mutual_Information.mp4 and DFC_{cond1_name.get()}_Mutual_Information.mp4 at\n"+os.getcwd())
                                else:
                                    showinfo(title="Info",message="Video of DFC were saved as\nDFC_{cond1_name.get()}_Mutual_Information.mp4 at\n"+os.getcwd())
                            winfilm.destroy()
                        btn2=Button(winfilm, text='Create video', command=process_frames)
                        btn2.grid(row=1,column=0,sticky=W,padx=10)
                        pbar2.grid(row=1,column=1,sticky=W)
                        pbtxt2.grid(row=1,column=2,sticky=W)
                    def save_corr():
                        showinfo(title="Info",message="The full results will be saved as a\nNumpy array of size (n_times,n_chans,n_chans))")
                        fname1 = fd.asksaveasfilename(title="Mutual Information DFC condition "+cond1_name.get(),defaultextension=".npy",filetypes=(("Numpy array", "*.npy"),("All Files", "*.*")))
                        np.save(fname1,dfc1)
                        if (x2 is not None) or (raw2 is not None):
                            fname2 = fd.asksaveasfilename(title="Mutual Information DFC condition "+cond2_name.get(),defaultextension=".npy",filetypes=(("Numpy array", "*.npy"),("All Files", "*.*")))
                            np.save(fname2,dfc2)
                    Button(win,text="Save results",command=save_corr).grid(row=19,column=0,padx=10,sticky=W)
                    Button(win,text="Make DFC animations",command=make_film).grid(row=20,column=0,padx=10,sticky=W)
                    Button(win,text="Close",command=win.destroy).grid(row=21,column=0,padx=10,pady=10,columnspan=5)
            else:
                showinfo(title="Error",message="Delay, window length and frame rate must be positive numbers")
        Label(win,text="Calculate mutual information").grid(row=0,column=0,columnspan=4,padx=10,pady=10)
        Label(win,text="Calculate from:").grid(row=1,column=0,padx=10,sticky=W)
        Radiobutton(win,text="From raw EEG",variable=raw_or_epoch,value=1).grid(row=1,column=1,sticky=W)
        Radiobutton(win,text="From epoch:",variable=raw_or_epoch,value=2).grid(row=2,column=1,sticky=W)
        Entry(win,textvariable=sel_epoch1,width=2).grid(row=2,column=2,sticky=W)
        Label(win,text=f"for {cond1_name.get()}").grid(row=2,column=3,sticky=W)
        Entry(win,textvariable=sel_epoch2,width=2).grid(row=3,column=2,sticky=W)
        Label(win,text=f"for {cond2_name.get()}").grid(row=3,column=3,sticky=W)
        Radiobutton(win,text="Average of results of all epochs",variable=raw_or_epoch,value=3).grid(row=4,column=1,sticky=W,columnspan=3)
        Label(win,text="Transmission delay between brain regions (ms):").grid(row=5,column=0,padx=10,sticky=W)
        Entry(win,textvariable=delay,width=4).grid(row=5,column=1,sticky=W)
        Label(win,text="Number of symbols (partition divisions):").grid(row=6,column=0,sticky=W,padx=10)
        Entry(win,textvariable=ns,width=3).grid(row=6,column=1,sticky=W)
        Label(win,text="Partition divisions:").grid(row=7,column=0,sticky=W,padx=10)
        Radiobutton(win,text="equal-sized divisions",variable=div_type,value=1).grid(row=7,column=1,sticky=W)
        Radiobutton(win,text="divisions with same number of points",variable=div_type,value=2).grid(row=8,column=1,columnspan=4,sticky=W)
        Radiobutton(win,text="other (separate with commas):",variable=div_type,value=3).grid(row=9,column=1,sticky=W)
        Label(win,text=" X divisions:").grid(row=9,column=2,sticky=W)
        Entry(win,textvariable=xdiv_vals,width=20).grid(row=9,column=3,sticky=W)
        Label(win,text=" Y divisions:").grid(row=10,column=2,sticky=W)
        Entry(win,textvariable=ydiv_vals,width=20).grid(row=10,column=3,sticky=W)
        Label(win,text="Symbolic length:").grid(row=11,column=0,sticky=W,padx=10)
        Label(win,text="for X:").grid(row=11,column=1,sticky=W)
        Entry(win,textvariable=lxp,width=3).grid(row=11,column=2,sticky=W)
        Label(win,text="for Y:").grid(row=12,column=1,sticky=W)
        Entry(win,textvariable=lyp,width=3).grid(row=12,column=2,sticky=W)
        Label(win,text="Tau:").grid(row=13,column=0,sticky=W,padx=10)
        Entry(win,textvariable=tau,width=3).grid(row=13,column=1,sticky=W)
        Label(win,text="Units:").grid(row=14,column=0,sticky=W,padx=10)
        OptionMenu(win,unit,*optionlist).grid(row=14,column=1,sticky=W)
        Label(win,text="Moving window length (ms):").grid(row=15,column=0,padx=10,sticky=W)
        Entry(win,textvariable=wlen,width=6).grid(row=15,column=1,sticky=W)
        Label(win,text="Moving window overlap (ms):").grid(row=16,column=0,padx=10,sticky=W)
        Entry(win,textvariable=woverlap,width=6).grid(row=16,column=1,sticky=W)
        Label(win,text="Frame rate:").grid(row=17,column=0,padx=10,sticky=W)
        Entry(win,textvariable=fr,width=6).grid(row=17,column=1,sticky=W)
        pbar=Progressbar(win,orient=HORIZONTAL,length=100,mode='determinate')
        pbar['value']=0.0
        pbtxt=Label(win,text="--")
        btn=Button(win, text='Start calculation', command=step)
        btn.grid(row=18,column=0,sticky=W,padx=10)
        pbar.grid(row=18,column=1,sticky=W)
        pbtxt.grid(row=18,column=2,sticky=W,columnspan=3)

def tfr():
    win=Toplevel(main)
    error=make_x()
    fmin=StringVar()
    fmin.set("5")
    fmax=StringVar()
    fmax.set("30")
    nfreq=StringVar()
    nfreq.set("50")
    if error==1:
        Label(win,text="ERROR\nIt is required at least\n1 preprocessed data",justify=CENTER).grid(row=0,column=0,padx=10,pady=10)        
        Button(win,text="OK",command=win.destroy).grid(row=1,column=0,padx=10)
    else:
        def calculate_tfr():
            if (float(fmin.get())>0) and (float(fmax.get())>float(fmin.get())) and (int(nfreq.get())>0):
                freqs=np.logspace(*np.log10([float(fmin.get()), float(fmax.get())]), num=int(nfreq.get()))
                n_cycles=freqs/2.
                power1=[]
                itc1=[]
                power2=[]
                itc2=[]
                pmax=len(event_dict.keys())
                if x2 is not None:
                    pmax=2*pmax
                k=0
                for key in event_dict.keys():
                    win.update_idletasks()
                    win.update()
                    k+=1
                    pbar['value'] += 100/pmax
                    pbtxt['text']=f"{k:d}/{pmax:d}"
                    a1,b1 = mne.time_frequency.tfr_morlet(x1[key], freqs=freqs, n_cycles=n_cycles, use_fft=True,return_itc=True, decim=3, n_jobs=1)
                    power1.append(a1)
                    itc1.append(b1)
                    if x2 is not None:
                        win.update_idletasks()
                        win.update()
                        k+=1
                        pbar['value'] += 100/pmax
                        pbtxt['text']=f"{k:d}/{pmax:d}"
                        a2,b2 = mne.time_frequency.tfr_morlet(x2[key], freqs=freqs, n_cycles=n_cycles, use_fft=True,return_itc=True, decim=3, n_jobs=1)
                        power2.append(a2)
                        itc2.append(b2)
                def make_plot():
                    optionlist=x1.ch_names
                    sel_chan=StringVar()
                    sel_chan.set(optionlist[0])
                    win2=Toplevel(win)
                    Label(win2,text="Plot TFR topoplot").grid(row=0,column=0,columnspan=5)
                    def make_topo(cond_name,key_name,cond_idx,key_idx):
                        if cond_idx==1:
                            power1[key_idx].plot_topo(mode='logratio', title='Average power, '+cond_name+", "+key_name)
                        else:
                            power2[key_idx].plot_topo(mode='logratio', title='Average power, '+cond_name+", "+key_name)
                    def make_topoitc(cond_name,key_name,cond_idx,key_idx):
                        if cond_idx==1:
                            itc1[key_idx].plot_topo(mode='logratio', title='Inter-trial coherence, '+cond_name+", "+key_name)
                        else:
                            itc2[key_idx].plot_topo(mode='logratio', title='Inter-trial coherence, '+cond_name+", "+key_name)
                    k=0    
                    for key in event_dict.keys():
                        Button(win2,text="Power topoplot "+cond1_name.get()+", "+key,command=partial(make_topo,cond1_name.get(),key,1,k)).grid(row=k+1,column=0)
                        Button(win2,text="ITC topoplot "+cond1_name.get()+", "+key,command=partial(make_topoitc,cond1_name.get(),key,1,k)).grid(row=k+1,column=1)
                        if x2 is not None:
                            Button(win2,text="Power topoplot "+cond2_name.get()+", "+key,command=partial(make_topo,cond2_name.get(),key,2,k)).grid(row=k+1,column=3)
                            Button(win2,text="ITC topoplot "+cond2_name.get()+", "+key,command=partial(make_topoitc,cond2_name.get(),key,2,k)).grid(row=k+1,column=4)
                        k+=1
                    if x2 is not None:
                        Separator(win2,orient="vertical").grid(row=1,column=2,rowspan=k,sticky='ns',padx=10)
                    Separator(win2,orient="horizontal").grid(row=k+2,column=0,columnspan=5,sticky='ew',pady=10)
                    Label(win2,text="Plot TFR for single channel").grid(row=k+3,column=0,columnspan=5)
                    Label(win2,text="Select channel: ").grid(row=k+4,column=0,sticky=W)
                    OptionMenu(win2,sel_chan,optionlist[0],*optionlist).grid(row=k+4,column=1,sticky=W)
                    def make_power(cond_name,key_name,cond_idx,key_idx,chann):
                        chanidx=x1.ch_names.index(chann.get())
                        if cond_idx==1:
                            power1[key_idx].plot([chanidx],mode='logratio', title='Average power, '+cond_name+", "+key_name+", "+power1[key_idx].ch_names[chanidx])
                        else:
                            power2[key_idx].plot([chanidx],mode='logratio', title='Average power, '+cond_name+", "+key_name+", "+power2[key_idx].ch_names[chanidx])
                    def make_itc(cond_name,key_name,cond_idx,key_idx,chann):
                        chanidx=x1.ch_names.index(chann.get())
                        if cond_idx==1:
                            itc1[key_idx].plot([chanidx],mode='logratio', title='Inter-trial coherence, '+cond_name+", "+key_name+", "+itc1[key_idx].ch_names[chanidx])
                        else:
                            itc2[key_idx].plot([chanidx],mode='logratio', title='Inter-trial coherence, '+cond_name+", "+key_name+", "+itc2[key_idx].ch_names[chanidx])
                    k2=0
                    for key in event_dict.keys():
                        Button(win2,text="Plot single channel power, "+cond1_name.get()+", "+key,command=partial(make_power,cond1_name.get(),key,1,k2,sel_chan)).grid(row=k+5+k2,column=0)
                        Button(win2,text="Plot single channel ITC, "+cond1_name.get()+", "+key,command=partial(make_itc,cond1_name.get(),key,1,k2,sel_chan)).grid(row=k+5+k2,column=1)
                        if x2 is not None:
                            Button(win2,text="Plot single channel power, "+cond2_name.get()+", "+key,command=partial(make_power,cond2_name.get(),key,2,k2,sel_chan)).grid(row=k+5+k2,column=3)
                            Button(win2,text="Plot single channel ITC, "+cond2_name.get()+", "+key,command=partial(make_itc,cond2_name.get(),key,2,k2,sel_chan)).grid(row=k+5+k2,column=4)
                        k2+=1
                    if x2 is not None:
                        Separator(win2,orient="vertical").grid(row=k+5,column=2,rowspan=k,sticky='ns',padx=10)
                    Separator(win2,orient="horizontal").grid(row=k+6+k2,column=0,columnspan=5,sticky='ew',pady=10)
                    Button(win2,text="Close",command=win2.destroy).grid(row=k+7+k2,column=0,columnspan=5)

                    
                def save_tfr():
                    showinfo(title="Info",message="The full results will be\nsaved as HDF-5 files")
                    k=0
                    for key in event_dict.keys():
                        fname1 = fd.asksaveasfilename(title="Time-frequency Power for condition "+cond1_name.get()+" event "+key,defaultextension=".npy",filetypes=(("HDF-5", "*.hdf5"),("All Files", "*.*")))
                        power1[k].save(fname1,overwrite=True)
                        fname1 = fd.asksaveasfilename(title="Time-frequency ITC for condition "+cond1_name.get()+" event "+key,defaultextension=".npy",filetypes=(("HDF-5", "*.hdf5"),("All Files", "*.*")))
                        itc1[k].save(fname1,overwrite=True)
                        if (x2 is not None) or (raw2 is not None):
                            fname2 = fd.asksaveasfilename(title="Time-frequency Power for condition "+cond2_name.get()+" event "+key,defaultextension=".npy",filetypes=(("HDF-5", "*.hdf5"),("All Files", "*.*")))
                            power2[k].save(fname2,overwrite=True)
                            fname2 = fd.asksaveasfilename(title="Time-frequency ITC for condition "+cond2_name.get()+" event "+key,defaultextension=".npy",filetypes=(("HDF-5", "*.hdf5"),("All Files", "*.*")))
                            itc2[k].save(fname2,overwrite=True)
                        k=k+1
                Button(win,text="Save results",command=save_tfr).grid(row=5,column=0,padx=10,sticky=W)
                Button(win,text="Plot",command=make_plot).grid(row=6,column=0,padx=10,sticky=W)
                Button(win,text="Close",command=win.destroy).grid(row=7,column=0,padx=10,pady=10,columnspan=5)
            else:
                showinfo(title="Error",message="Frequencies and number of boxes must be greater than zero,\nand maximum frequency must be greater than minimum frequency.")
        Label(win,text="Calculate Time-Frequency Power and Inter-Trial Coherence from Epochs").grid(row=0,column=0,columnspan=4,padx=10,pady=10)
        Label(win,text="Minimum frequency (Hz)").grid(row=1,column=0,padx=10,sticky=W)
        Entry(win,textvariable=fmin,width=4).grid(row=1,column=1,sticky=W)
        Label(win,text="Maximum frequency (Hz)").grid(row=2,column=0,padx=10,sticky=W)
        Entry(win,textvariable=fmax,width=4).grid(row=2,column=1,sticky=W)
        Label(win,text="Number of frequency boxes").grid(row=3,column=0,padx=10,sticky=W)
        Entry(win,textvariable=nfreq,width=4).grid(row=3,column=1,sticky=W)
        btn=Button(win,text="Calculate",command=calculate_tfr)
        pbar=Progressbar(win,orient=HORIZONTAL,length=100,mode='determinate')
        pbar['value']=0.0
        pbtxt=Label(win,text="--")
        btn.grid(row=4,column=0,padx=10,sticky=W)
        pbar.grid(row=4,column=1,sticky=W)
        pbtxt.grid(row=4,column=2,sticky=W,columnspan=3)

def animtopo():
    win=Toplevel(main)
    global x1
    global x2
    error=make_x()
    if error==1:
        Label(win,text="ERROR\nIt is required at least\n1 preprocessed data",justify=CENTER).grid(row=0,column=0,padx=10,pady=10)        
        Button(win,text="OK",command=win.destroy).grid(row=1,column=0,padx=10)
    else:
        framerate=StringVar()
        framerate.set("30")
        mean_or_indiv=IntVar()
        mean_or_indiv.set(1)
        Label(win,text="Generate animated topoplots:").grid(row=0,column=0,padx=10,pady=10)
        Radiobutton(win,text="For average of epochs",variable=mean_or_indiv,value=1).grid(row=0,column=1,pady=10,sticky=W,columnspan=3)
        Radiobutton(win,text="For individual epochs",variable=mean_or_indiv,value=2).grid(row=1,column=1,pady=10,sticky=W,columnspan=3)
        Label(win,text="Frame rate: ").grid(row=2,column=0,padx=0,sticky=W)
        Entry(win,textvariable=framerate,width=3).grid(row=2,column=1,sticky=W)
        def proceed():
            fr=int(framerate.get())
            win.destroy()
            if mean_or_indiv.get()==2:
                for key in event_dict.keys():
                    fname1 = fd.asksaveasfilename(title="Animated topoplot condition "+cond1_name.get()+" event "+key,defaultextension=".mp4",filetypes=(("MPEG-4", "*.mp4"),("All Files", "*.*")))
                    win2=Toplevel(main)
                    lab=Label(win2,text="Generating animations, may take a while\n(Please do not close the program,\nit is not frozen)")
                    lab.grid(row=0,column=0,padx=10,pady=10)
                    win2.update()
                    n_epochs=len(x1[key])
                    for epoch in range(n_epochs):
                        lab['text']=f'Generating animations, may take a while\nepoch {epoch+1} of {n_epochs}\n(Please do not close the program,\nit is not frozen)'
                        win2.update()
                        evoked=mne.EvokedArray(x1[key].get_data()[epoch,:,:],x1.info)
                        fig,anim=evoked.animate_topomap(blit=False,show=False,frame_rate=fr,times=x1.times[1:-1]-x1.times[0])
                        anim.save(f'{fname1[:-4]}_{epoch:03}.mp4')
                        fig.clf()
                        plt.close()
                        gc.collect()
                    win2.destroy()
                    if x2 is not None:
                        fname2 = fd.asksaveasfilename(title="Animated topoplot condition "+cond2_name.get()+" event "+key,defaultextension=".mp4",filetypes=(("MPEG-4", "*.mp4"),("All Files", "*.*")))
                        win2=Toplevel(main)
                        lab=Label(win2,text="Generating animations, may take a while\n(Please do not close the program,\nit is not frozen)")
                        lab.grid(row=0,column=0,padx=10,pady=10)
                        win2.update()
                        n_epochs=len(x2[key])
                        for epoch in range(n_epochs):
                            lab['text']=f'Generating animations, may take a while\nepoch {epoch+1} of {n_epochs}\n(Please do not close the program,\nit is not frozen)'
                            win2.update()
                            evoked=mne.EvokedArray(x2[key].get_data()[epoch,:,:],x2.info)
                            fig,anim=evoked.animate_topomap(blit=False,show=False,frame_rate=fr,times=x2.times[1:-1]-x2.times[0])
                            anim.save(f'{fname2[:-4]}_{epoch:03}.mp4')
                            fig.clf()
                            plt.close()
                            gc.collect()
                        win2.destroy()
            else:
                for key in event_dict.keys():
                    fname1 = fd.asksaveasfilename(title="Animated topoplot condition "+cond1_name.get()+" event "+key,defaultextension=".mp4",filetypes=(("MPEG-4", "*.mp4"),("All Files", "*.*")))
                    win2=Toplevel(main)
                    Label(win2,text="Calculating averages, please wait\n(Please do not close the program,\nit is not frozen)").grid(row=0,column=0,padx=10,pady=10)
                    win2.update()
                    mean_x1=np.zeros(x1[key].get_data().shape[1:])
                    for ch in range(x1[key].get_data().shape[1]):
                        for t in range(x1[key].get_data().shape[2]):
                            mean_x1[ch,t]=np.nanmean(x1[key].get_data()[:,ch,t])
                    win2.destroy()
                    win2=Toplevel(main)
                    Label(win2,text="Generating animations, may take a while\n(Please do not close the program,\nit is not frozen)").grid(row=0,column=0,padx=10,pady=10)
                    win2.update()
                    evoked=mne.EvokedArray(mean_x1,x1.info)
                    fig,anim=evoked.animate_topomap(blit=False,show=False,frame_rate=fr,times=x1.times[1:-1]-x1.times[0])
                    anim.save(fname1)
                    fig.clf()
                    plt.close()
                    gc.collect()
                    win2.destroy()
                    if x2 is not None:
                        fname2 = fd.asksaveasfilename(title="Animated topoplot condition "+cond2_name.get()+" event "+key,defaultextension=".mp4",filetypes=(("MPEG-4", "*.mp4"),("All Files", "*.*")))
                        win2=Toplevel(main)
                        Label(win2,text="Calculating averages, please wait\n(Please do not close the program,\nit is not frozen)").grid(row=0,column=0,padx=10,pady=10)
                        win2.update()
                        mean_x2=np.zeros(x2[key].get_data().shape[1:])
                        for ch in range(x2[key].get_data().shape[1]):
                            for t in range(x2[key].get_data().shape[2]):
                                mean_x2[ch,t]=np.nanmean(x2[key].get_data()[:,ch,t])
                        win2.destroy()
                        win2=Toplevel(main)
                        Label(win2,text="Generating animations, may take a while\n(Please do not close the program,\nit is not frozen)").grid(row=0,column=0,padx=10,pady=10)
                        win2.update()
                        evoked=mne.EvokedArray(mean_x2,x2.info)
                        fig,anim=evoked.animate_topomap(blit=False,show=False,frame_rate=fr,times=x2.times[1:-1]-x2.times[0])
                        anim.save(fname2)
                        fig.clf()
                        plt.close()
                        gc.collect()
                        win2.destroy()
        Button(win,text="OK",command=proceed).grid(row=3,column=0,padx=10,pady=10,columnspan=3)

#Find optimal Delay
def complexity_delay(signal, delay_max=None):
    """Automated selection of the optimal Time Delay (tau) for time-delay embedding.
    Rosenstein (1993) suggests to approximate the point where the autocorrelation function drops
    to (1 − 1 / e) of its maximum value.
    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay_max : int
        The maximum time delay (Tau or lag) to test.
    Returns
    -------
    delay : int
        Optimal time delay.
    """
    # Initalize vectors
    if delay_max is None:
        delay_max=min([len(signal)/4,1000])    
    delay_max=int(delay_max)
    tau_sequence = np.arange(delay_max)
    # Get metric
    def autocor(x):
        a = np.concatenate((x, np.zeros(len(signal) - 1)))
        A = np.fft.fft(a)
        S = np.conj(A) * A
        c_fourier = np.fft.ifft(S)
        acov = c_fourier[: (c_fourier.size // 2) + 1].real
        r = acov / acov[0]
        return r
    metric_values = autocor(signal)[:len(tau_sequence)]
    # Get optimal tau
    optimal_tau = np.where(metric_values < np.max(metric_values) * (1 - 1.0 / np.e))[0]
    if optimal_tau.size==0:
        return np.nan
    elif optimal_tau.size>1:
        optimal_tau=optimal_tau[0]
    optimal_tau = tau_sequence[optimal_tau]
    # Return optimal tau
    return optimal_tau

#Find optimal embedding
def optimal_dimension(signal, delay=1, dimension_max=None, **kwargs):
    """Automated selection of the optimal Dimension (m) for time-delay embedding,
    using false nearest neighbors (FNN) method described by Kennel et al (1992).
    Returns the minimum dimension and a dictionary with the vectors:
    - f1 : Fraction of neighbors classified as false by Test I.
    - f2 : Fraction of neighbors classified as false by Test II.
    - f3 : Fraction of neighbors classified as false by either Test I or Test II.
    """
    try:
        delay=int(delay)
    except:
        delay=1
    if dimension_max is None:
        for dim in range(1,20):
            if (len(signal[:-delay])-(dim-1)*delay)<=0:
                dimension_max=dim
                break
            else:
                dimension_max=20
    # Initialize vectors
    dimension_seq = np.arange(1,dimension_max)
    # Method
    def _embedding_dimension_neighbors(y, metric="euclidean", window=0, maxnum=None):
        """Find nearest neighbors of all points in the given array. Finds the nearest neighbors of all points in the
        given array using SciPy's KDTree search.
        """
        p = 2
        tree = spatial.cKDTree(y)
        n = len(y)
        if not maxnum:
            maxnum = (window + 1) + 1 + (window + 1)
        else:
            maxnum = max(1, maxnum)
        if maxnum >= n:
            raise ValueError("maxnum is bigger than array length.")
        # Query for k numbers of nearest neighbors
        distances, indices = tree.query(y, k=range(1, maxnum + 2), p=p)
        # Substract the first point
        valid = indices - np.tile(np.arange(n), (indices.shape[1], 1)).T
        # Remove points that are closer than min temporal separation
        valid = np.abs(valid) > window
        # Remove also self reference (d > 0)
        valid = valid & (distances > 0)
        # Get indices to keep
        valid = (np.arange(len(distances)), np.argmax(valid, axis=1))
        distances = distances[valid]
        indices = indices[(valid)]
        return indices, distances
    def _embedding_dimension_d(signal, dimension, delay=1, window=10, maxnum=None):
        """We need to reduce the number of points in dimension d by tau
        so that after reconstruction, there'll be equal number of points
        at both dimension d as well as dimension d + 1."""
        y1 = takens_embedding(signal[:-delay], delay=delay, dimension=dimension)
        y2 = takens_embedding(signal, delay=delay, dimension=dimension + 1)
        # Find near neighbors in dimension d.
        index, dist = _embedding_dimension_neighbors(y1, window=window, maxnum=maxnum)
        # Compute the near-neighbor distances in d + 1 dimension
        d = [spatial.distance.chebyshev(i, j) for i, j in zip(y2, y2[index])]
        return np.asarray(d), dist, index, y2
    def _embedding_dimension_ffn_d(signal, dimension, delay=1, R=10.0, A=2.0, metric="euclidean", window=10, maxnum=None):
        """Return fraction of false nearest neighbors for a single d."""
        d, dist, index, y2 = _embedding_dimension_d(signal, dimension, delay, window, maxnum)
        # Find all potential false neighbors using Kennel et al.'s tests.
        dist[dist == 0] = np.nan  # assign nan to avoid divide by zero error in next line
        f1 = np.abs(y2[:, -1] - y2[index, -1]) / dist > R
        f2 = d / np.std(signal) > A
        f3 = f1 | f2
        return np.mean(f1), np.mean(f2), np.mean(f3)
    def _embedding_dimension_ffn(signal, dimension_seq, delay=1, R=10.0, A=2.0, **kwargs):
        """Compute the fraction of false nearest neighbors f1,f2,f3"""
        values = np.asarray([_embedding_dimension_ffn_d(signal, dimension, delay, R=R, A=A, **kwargs) for dimension in dimension_seq]).T
        f1, f2, f3 = values[0, :], values[1, :], values[2, :]
        return f1, f2, f3
    f1, f2, f3 = _embedding_dimension_ffn(signal, dimension_seq=dimension_seq, delay=delay, **kwargs)
    min_dimension = [i for i, x in enumerate(f3 <= 1.85 * np.min(f3[np.nonzero(f3)])) if x][0]
    # Store information
    info = {"Method": "False Nearest Neighbours", "Values": dimension_seq, "f1": f1, "f2": f2, "f3": f3}
    return min_dimension, info

#Takens Embedding
def takens_embedding(signal, delay=1, dimension=3, show=False, **kwargs):
    """Time-delay embedding of a signal, according to Takens'(1981) theorem"""
    N = len(signal)
    Y = np.zeros((dimension, N - (dimension - 1) * delay))
    for i in range(dimension):
        Y[i] = signal[i * delay : i * delay + Y.shape[1]]
    embedded = Y.T
    return embedded

#Maximum Lyapunov Exponent
def complexity_lyapunov(signal,delay=None,dimension=None,len_trajectory=20,min_neighbors="default",fs=1000):
    """(Largest) Lyapunov Exponent (LLE)
    Lyapunov exponents (LE) describe the rate of exponential separation (convergence or divergence)
    of nearby trajectories of a dynamical system. The largest LE value, `LLE` is often used to
    determine the overall predictability of the dynamical system. Method: Rosenstein1993
    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    delay : int, None
        Time delay (often denoted 'Tau', sometimes referred to as 'lag').If None, the delay is set to
        distance where the autocorrelation function drops below 1 - 1/e times its original value.
    dimension : int
        Embedding dimension (often denoted 'm' or 'd', sometimes referred to as 'order').
    len_trajectory : int
        The number of data points in which neighbouring trajectories are followed.
    fs: int
        Sampling rate of the data
    Returns
    --------
    lle : float
        An estimate of the largest Lyapunov exponent (LLE)
    """
    # Sanity checks
    if isinstance(signal, (np.ndarray, pd.DataFrame)) and signal.ndim > 1:
        raise ValueError("Incorrect signal shape")
    # Find tolerance (separation between neighbours)
    psd = np.abs(np.fft.rfft(signal))** 2
    frequency = np.linspace(0,fs/2,len(psd))
    mean_freq = np.sum(psd*frequency)/np.sum(psd)
    mean_period = 1/mean_freq  # seconds per cycle
    tolerance = int(np.ceil(mean_period*fs))
    # Method
    # Delay embedding
    if delay is None:
        delay = complexity_delay(signal)
    #optimal dimension
    if dimension is None:
        dimension,_=optimal_dimension(signal, delay=delay, dimension_max=None)
    # Check that sufficient data points are available
    min_len = (dimension - 1) * delay + 1 # min len for single orbit vector
    min_len += len_trajectory - 1 # min len for orbit vectors following a complete trajectory
    min_len += tolerance * 2 + 1 # need tolerance * 2 + 1 orbit vectors to find neighbors for each
    if len(signal) < min_len:
        raise ValueError(f"Error: for dimension={dimension}, delay={delay}, tolerance={tolerance} and " + f"len_trajectory={len_trajectory}, you need at least {min_len} datapoints in your time series, but you have {len(signal)}.")
    # Embed
    embedded = takens_embedding(signal,delay=delay,dimension=dimension)
    m = len(embedded)
    # Construct matrix with pairwise distances between vectors in orbit
    dists = euclidean_distances(embedded)
    for i in range(m):
        # Exclude indices within tolerance
        dists[i, max(0, i - tolerance) : i + tolerance + 1] = np.inf
    # Find indices of nearest neighbours
    ntraj = m - len_trajectory + 1
    min_dist_indices = np.argmin(dists[:ntraj, :ntraj], axis=1)  # exclude last few indices
    min_dist_indices = min_dist_indices.astype(int)
    # Follow trajectories of neighbour pairs for len_trajectory data points
    trajectories = np.zeros(len_trajectory)
    for k in range(len_trajectory):
        divergence = dists[(np.arange(ntraj) + k, min_dist_indices + k)]
        dist_nonzero = np.where(divergence != 0)[0]
        if len(dist_nonzero) == 0:
            trajectories[k] = -np.inf
        else:
            # Get average distances of neighbour pairs along the trajectory
            trajectories[k] = np.mean(np.log(divergence[dist_nonzero]))
    divergence_rate = trajectories[np.isfinite(trajectories)]
    # LLE obtained by least-squares fit to average line
    le, _ = np.polyfit(np.arange(1,len(divergence_rate)+1),divergence_rate,1)
    return le

def lyapunov():
    error=make_x()
    win=Toplevel(main)
    if error==1:
        Label(win,text="ERROR\nIt is required at least\n1 preprocessed data",justify=CENTER).grid(row=0,column=0,padx=10,pady=10)        
        Button(win,text="OK",command=win.destroy).grid(row=1,column=0,padx=10)
    else:
        sel_chan_name=StringVar()
        sel_tau=StringVar()
        sel_tau.set("1")
        sel_dim=StringVar()
        sel_dim.set("2")
        namelist=eeg1.ch_names
        sel_chan_name.set(namelist[0])
        def find_optimal_params():
            event_names=list(event_dict.keys())
            optim_tau=np.empty((1,len(event_names),len(x1.events)+1))
            optim_dim=np.empty((1,len(event_names),len(x1.events)+1))
            if x2 is not None:
                optim_tau=np.empty((2,len(event_names),len(x1.events)+1))
                optim_dim=np.empty((2,len(event_names),len(x1.events)+1))
            optim_tau.fill(np.nan)
            optim_dim.fill(np.nan)
            count=0
            pmax2=0
            for k in range(len(event_names)):
                for ev in range(len(x1[event_names[k]])):
                    pmax2+=1
                if x2 is not None:
                    for ev in range(len(x2[event_names[k]])):
                        pmax2+=1
            for k in range(len(event_names)):
                vals1=x1[event_names[k]].get_data()
                for ev in range(vals1.shape[0]):
                    count+=1
                    win.update_idletasks()
                    win.update()
                    pbar2['value'] += 100/pmax2
                    pbtxt2['text']=f"{count:d}/{pmax2:d}"
                    optim_tau[0,k,ev]=int(complexity_delay(vals1[ev,sel_chan,:]))
                    optim_dim[0,k,ev],_=optimal_dimension(vals1[ev,sel_chan,:], delay=optim_tau[0,k,ev], dimension_max=None)
                if x2 is not None:
                    vals2=x2[event_names[k]].get_data()
                    for ev in range(vals2.shape[0]):
                        count+=1
                        win.update_idletasks()
                        win.update()
                        pbar2['value'] += 100/pmax2
                        pbtxt2['text']=f"{count:d}/{pmax2:d}"
                        optim_tau[1,k,ev]=int(complexity_delay(vals2[ev,sel_chan,:]))
                        optim_dim[1,k,ev],_=optimal_dimension(vals2[ev,sel_chan,:], delay=optim_tau[0,k,ev], dimension_max=None)
            cond_names=[cond1_name.get()]
            if x2 is not None:
                cond_names.append(cond2_name.get())
            titles=[]
            rows_tau=[]
            rows_dim=[]
            for cond in range(len(cond_names)):
                for ev in range(len(event_names)):
                    titles.append(cond_names[cond]+'\n'+event_names[ev])
                    rows_tau.append(optim_tau[cond,ev,:])
                    rows_dim.append(optim_dim[cond,ev,:])
            tau_df=pd.DataFrame(data=rows_tau,index=titles).T
            dim_df=pd.DataFrame(data=rows_dim,index=titles).T
            plt.figure(1)
            ax = sns.boxplot(data=tau_df,orient="v")
            ax = sns.swarmplot(data=tau_df,orient="v",color=".7")
            ax.set_xlabel("Condition/event")
            ax.set_ylabel("Optimal tau (Takens' reconstruction delay)")
            ax.set_title(sel_chan_name.get())
            plt.figure(2)
            ax2 = sns.boxplot(data=dim_df,orient="v")
            ax2 = sns.swarmplot(data=dim_df,orient="v",color=".7")
            ax2.set_xlabel("Condition/event")
            ax2.set_ylabel("Optimal reconstruction dimension")
            ax2.set_title(sel_chan_name.get())
            plt.show()                
        def calculate_lyap():
            k=0
            count=0
            event_names=list(event_dict.keys())
            le=np.empty((1,len(event_dict.keys()),len(x1.events)))
            le.fill(np.nan)
            if x2 is not None:
                le=np.empty((2,len(event_dict.keys()),max((len(x1.events),len(x2.events)))))
                le.fill(np.nan)                            
            pmax=len(event_dict.keys())
            if x2 is not None:
                pmax=len(event_dict.keys())*2
            for key in event_dict.keys():
                k+=1
                count+=1
                win.update_idletasks()
                win.update()
                pbar['value'] += 100/pmax
                pbtxt['text']=f"{count:d}/{pmax:d}"
                #compute lyapunov
                vals1=x1[event_names[k-1]].get_data()
                for ev in range(vals1.shape[0]):
                    le[0,k-1,ev]=complexity_lyapunov(vals1[ev,sel_chan,:],delay=int(sel_tau.get()),dimension=int(sel_dim.get()),len_trajectory=20,min_neighbors="default",fs=x1.info["sfreq"])
                if x2 is not None:
                    win.update_idletasks()
                    win.update()
                    count+=1
                    pbar['value'] += 100/pmax
                    pbtxt['text']=f"{count:d}/{pmax:d}"
                    #compute lyapunov
                    vals2=x2[event_names[k-1]].get_data()
                    for ev in range(vals2.shape[0]):
                        le[1,k-1,ev]=complexity_lyapunov(vals2[ev,sel_chan,:],delay=int(sel_tau.get()),dimension=int(sel_dim.get()),len_trajectory=20,min_neighbors="default",fs=x1.info["sfreq"])
            cond_names=[cond1_name.get()]
            if x2 is not None:
                cond_names.append(cond2_name.get())
            titles=[]
            rows=[]
            for cond in range(le.shape[0]):
                for ev in range(len(event_names)):
                    titles.append(cond_names[cond]+'\n'+event_names[ev])
                    rows.append(le[cond,ev,:])
            le_df=pd.DataFrame(data=rows,index=titles).T
            plt.figure()
            ax = sns.boxplot(data=le_df,orient="v")
            ax = sns.swarmplot(data=le_df,orient="v",color=".7")
            ax.set_xlabel("Condition/event")
            ax.set_ylabel("Maximum Lyapunov Exponent")
            ax.set_title(sel_chan_name.get())
            plt.show()
            def save_lyap():
                fname = fd.asksaveasfilename(title="Save Lyapunov exponents channel "+sel_chan_name.get(),defaultextension=".csv",filetypes=(("Comma separated values (csv)", "*.csv"),("All Files", "*.*")))
                le_df.to_csv(fname)
            Button(win,text="Save results",command=save_lyap).grid(row=6,column=0,padx=10,sticky=W)
            Button(win,text="Close",command=win.destroy).grid(row=7,column=0,padx=10,pady=10,columnspan=5)
        Label(win,text="Calculate Maximum Lyapunov Exponent").grid(row=0,column=0,columnspan=4,padx=10,pady=10)
        Label(win,text="Select channel").grid(row=1,column=0,padx=10,sticky=W)
        OptionMenu(win,sel_chan_name,namelist[0],*namelist).grid(row=1,column=1,padx=10,sticky=W)
        sel_chan=namelist.index(sel_chan_name.get())
        #Entry(win,textvariable=sel_chan,width=4).grid(row=1,column=1,sticky=W)
        btn_find_optim=Button(win,text="Find optimal parameters\n(tau/dimension)",command=find_optimal_params)
        pbar2=Progressbar(win,orient=HORIZONTAL,length=100,mode='determinate')
        pbar2['value']=0.0
        pbtxt2=Label(win,text="--")
        Label(win,text="Select tau").grid(row=3,column=0,padx=10,sticky=W)
        Entry(win,textvariable=sel_tau,width=4).grid(row=3,column=1,sticky=W)
        Label(win,text="Select dimension").grid(row=4,column=0,padx=10,sticky=W)
        Entry(win,textvariable=sel_dim,width=4).grid(row=4,column=1,sticky=W)
        btn=Button(win,text="Calculate",command=calculate_lyap)
        pbar=Progressbar(win,orient=HORIZONTAL,length=100,mode='determinate')
        pbar['value']=0.0
        pbtxt=Label(win,text="--")
        btn_find_optim.grid(row=2,column=0,padx=10,sticky=W)
        pbar2.grid(row=2,column=1,sticky=W)
        pbtxt2.grid(row=2,column=2,sticky=W,columnspan=3)
        btn.grid(row=5,column=0,padx=10,sticky=W)
        pbar.grid(row=5,column=1,sticky=W)
        pbtxt.grid(row=5,column=2,sticky=W,columnspan=3)


main=Tk()
main.title("EEG Causality Tools")

raw_or_energy=IntVar()
raw_or_energy.set(1)
cond1_name=StringVar()
cond1_name.set("DBS Off")
cond2_name=StringVar()
cond2_name.set("DBS On")


#main window layout
Label(main,text="EEG \nCausality \nTools").grid(row=0,column=0,padx=(10,0),pady=(10,0),rowspan=15,sticky=W)
Separator(main,orient="vertical").grid(row=0,column=1,rowspan=8,sticky='ns')
Separator(main,orient="horizontal").grid(row=0,column=1,columnspan=8,sticky='ew')
Label(main,text="Load data").grid(row=1,column=2,columnspan=2)
Button(main,text="Load raw EEG (*.edf)",command=load_edf,width=22).grid(row=2,column=2,padx=10,pady=(10,0),sticky=W)
Button(main,text="Load/select EEG montage",command=load_montage,width=22).grid(row=2,column=3,padx=10,pady=(10,0),sticky=E)
Button(main,text="Preprocess raw EEG",command=preprocess,width=22).grid(row=3,column=2,padx=10,pady=10,sticky=W)
frame_cond_1=Frame(main)
Label(frame_cond_1,text="Condition 1 name:").grid(row=0,column=0,sticky=W)
Entry(frame_cond_1,textvariable=cond1_name,width=8).grid(row=0,column=1,sticky=E)
frame_cond_1.grid(row=3,column=3,padx=10,pady=10)
Button(main,text="Load preprocessed EEG (*.fif)",command=load_fif,width=22).grid(row=4,column=2,padx=10,sticky=W)
frame_cond_2=Frame(main)
Label(frame_cond_2,text="Condition 2 name:").grid(row=0,column=0,sticky=W)
Entry(frame_cond_2,textvariable=cond2_name,width=8).grid(row=0,column=1,sticky=E)
frame_cond_2.grid(row=4,column=3,padx=10,pady=10)
Button(main,text="Select frequency band",command=select_freq,width=22).grid(row=5,column=2,padx=10,pady=10,sticky=W)
Radiobutton(main,text="Analyse using\nraw value (\u00B5V)",variable=raw_or_energy,value=1).grid(row=6,column=2,rowspan=2,padx=10,pady=10,sticky=W)
Radiobutton(main,text="Analyse using\nenergy value ((\u00B5V)\u00B2)",variable=raw_or_energy,value=2).grid(row=6,column=3,rowspan=2,padx=10,pady=10,sticky=W)
Separator(main,orient="vertical").grid(row=0,column=4,rowspan=8,sticky='ns')
Separator(main,orient="horizontal").grid(row=8,column=1,columnspan=4,sticky='ew')
Label(main,text="Correlation analysis on epochs").grid(row=1,column=5,columnspan=2)
Button(main,text="Pearson correlation",command=pearson_corr,width=22).grid(row=2,column=5,padx=10,pady=10,sticky=W)
Button(main,text="Spearman correlation",command=spearman_corr,width=22).grid(row=2,column=6,padx=10,pady=10,sticky=E)
Separator(main,orient="vertical").grid(row=0,column=7,rowspan=8,sticky='ns')
Separator(main,orient="horizontal").grid(row=3,column=4,columnspan=4,sticky='new')
Label(main,text="Frequency analysis on epochs").grid(row=3,column=5,columnspan=2)
Button(main,text="Coherence",command=coherence,width=22).grid(row=4,column=5,padx=10,pady=10,sticky=W)
Button(main,text="Weighted phase\n      lag index",command=wpli,width=22).grid(row=4,column=6,padx=10,pady=10,sticky=E)
Separator(main,orient="horizontal").grid(row=5,column=4,columnspan=4,sticky='new')
Label(main,text="Information-theory analysis on epochs").grid(row=5,column=5,columnspan=2)
Button(main,text="Mutual information",command=mi,width=22).grid(row=6,column=5,padx=10,pady=10,sticky=W)
Button(main,text="Transfer entropy",command=te,width=22).grid(row=6,column=6,padx=10,pady=10,sticky=E)
Separator(main,orient="horizontal").grid(row=8,column=4,columnspan=4,sticky='new')
Separator(main,orient="vertical").grid(row=9,column=1,rowspan=5,sticky='ns')
Label(main,text="Build dynamic functional connectome\non whole EEG time-series",justify=CENTER).grid(row=9,column=2,columnspan=2,rowspan=2,pady=(10,0))
Button(main,text="Pearson correlation",command=pearson_dfc,width=22).grid(row=11,column=2,padx=10,pady=10,sticky=W)
Button(main,text="Spearman correlation",command=spearman_dfc,width=22).grid(row=11,column=3,padx=10,pady=10,sticky=E)
Button(main,text="Mutual information",command=mi_dfc,width=22).grid(row=12,column=2,padx=10,pady=10,sticky=W)
Button(main,text="Transfer entropy",command=te_dfc,width=22).grid(row=12,column=3,padx=10,pady=10,sticky=E)
Separator(main,orient="vertical").grid(row=9,column=4,rowspan=5,sticky='ns')
Separator(main,orient="horizontal").grid(row=14,column=1,columnspan=4,sticky='ew')
Label(main,text="Additional tools").grid(row=9,column=5,pady=10,columnspan=2)
Button(main,text="Time-frequency analysis",command=tfr,width=22).grid(row=11,column=5,pady=10,padx=10)
Button(main,text="Animated topoplot",command=animtopo,width=22).grid(row=11,column=6,pady=10,padx=10)
Button(main,text="Lyapunov exponent",command=lyapunov,width=22).grid(row=12,column=5,pady=10,padx=10)
Separator(main,orient="vertical").grid(row=9,column=7,rowspan=5,sticky='ns')
Separator(main,orient="horizontal").grid(row=14,column=5,columnspan=4,sticky='ew')

#run
main.mainloop()
