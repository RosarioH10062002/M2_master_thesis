import mne
from pathlib import Path
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter
from mne.time_frequency import tfr_array_morlet
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu
import seaborn as sns
from mne_icalabel import label_components
from io import StringIO
from scipy.signal import find_peaks

mne_files_path = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Raw_data_eeg_psychopy_trimmed")
mne_root_epochs = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Epoch_files")
info_datasets = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Important_datasets\Bitbrain_eeg_sessions.csv")
segments_path = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Segments")
results_EEG_path = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Results")
dataframe_results = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Results\eeg_dataframe.csv")

final_dataframe_results = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Results\behavior_eeg.csv") #Final dataframe with everything of the power
final_dataframe_behavior_per_trials = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Important_datasets\dataset_trials_all_ids.csv") # Dataset with every trial 

def clean_dataframe_results(dataset):
    dataset = dataset.drop(
        columns=[
            "WEEK", "DATE", "IT", "DF", "BRIGHTNESS", "F_ORDER",
            "DPrime_Blocks",
            "Accuracy_Go", "Accuracy_NoGo",
            "Variability_Go", "Variability_NoGo",
            "Variability_Overall", "MeanGoRT"
        ]
    )

    dataset = dataset.rename(columns={

        # --
        "EEG_PRE_OCC_Delta_rel": "PRE_OCC_Delta",
        "EEG_PRE_OCC_Theta_rel": "PRE_OCC_Theta",
        "EEG_PRE_OCC_Alpha_rel": "PRE_OCC_Alpha",
        "EEG_PRE_OCC_Beta_rel": "PRE_OCC_Beta",
        "EEG_PRE_OCC_Gamma_rel": "PRE_OCC_Gamma",
        "EEG_PRE_OCC_Theta_Beta": "PRE_OCC_TB",

        # --
        "EEG_PRE_FRONT_Delta_rel": "PRE_FRONT_Delta",
        "EEG_PRE_FRONT_Theta_rel": "PRE_FRONT_Theta",
        "EEG_PRE_FRONT_Alpha_rel": "PRE_FRONT_Alpha",
        "EEG_PRE_FRONT_Beta_rel": "PRE_FRONT_Beta",
        "EEG_PRE_FRONT_Gamma_rel": "PRE_FRONT_Gamma",
        "EEG_PRE_FRONT_Theta_Beta": "PRE_FRONT_TB",

        # ---
        "EEG_POST_OCC_Delta_rel": "POST_OCC_Delta",
        "EEG_POST_OCC_Theta_rel": "POST_OCC_Theta",
        "EEG_POST_OCC_Alpha_rel": "POST_OCC_Alpha",
        "EEG_POST_OCC_Beta_rel": "POST_OCC_Beta",
        "EEG_POST_OCC_Gamma_rel": "POST_OCC_Gamma",
        "EEG_POST_OCC_Theta_Beta": "POST_OCC_TB",

        # --
        "EEG_POST_FRONT_Delta_rel": "POST_FRONT_Delta",
        "EEG_POST_FRONT_Theta_rel": "POST_FRONT_Theta",
        "EEG_POST_FRONT_Alpha_rel": "POST_FRONT_Alpha",
        "EEG_POST_FRONT_Beta_rel": "POST_FRONT_Beta",
        "EEG_POST_FRONT_Gamma_rel": "POST_FRONT_Gamma",
        "EEG_POST_FRONT_Theta_Beta": "POST_FRONT_TB",

        # ---
        "EEG_DELTA_OCC_Delta_rel": "DELTA_OCC_Delta",
        "EEG_DELTA_OCC_Theta_rel": "DELTA_OCC_Theta",
        "EEG_DELTA_OCC_Alpha_rel": "DELTA_OCC_Alpha",
        "EEG_DELTA_OCC_Beta_rel": "DELTA_OCC_Beta",
        "EEG_DELTA_OCC_Gamma_rel": "DELTA_OCC_Gamma",
        "EEG_DELTA_OCC_Theta_Beta": "DELTA_OCC_TB",

        # ---
        "EEG_DELTA_FRONT_Delta_rel": "DELTA_FRONT_Delta",
        "EEG_DELTA_FRONT_Theta_rel": "DELTA_FRONT_Theta",
        "EEG_DELTA_FRONT_Alpha_rel": "DELTA_FRONT_Alpha",
        "EEG_DELTA_FRONT_Beta_rel": "DELTA_FRONT_Beta",
        "EEG_DELTA_FRONT_Gamma_rel": "DELTA_FRONT_Gamma",
        "EEG_DELTA_FRONT_Theta_Beta": "DELTA_FRONT_TB",
    })
    return dataset  

BAD_CHANNELS = {
    (2, "27-05-26", "A"): "AF7,P8",
    (5, "13-05-26", "A"): "AF7",
    (6, "11-05-26", "B"): "O2",
    (6, "13-05-26", "B"): "O1,P7",
    (6, "19-05-26", "B"): "O1,P7,P8,O2",
    (8, "12-05-26", "A"): "AF8",
    (8, "13-05-26", "A"): "P7",
    (8, "18-05-26", "A"): "O1,P7,P8,O2",
    (10, "26-05-26", "A"): "Fp2",
    (10, "28-05-26", "A"): "Fp2",
    (10, "15-06-26", "B"): "P8",
    (13, "29-05-26", "B"): "P8",
    (13, "17-06-26", "B"): "P7",
    (13, "24-06-26", "B"): "P7",
    (13, "22-06-26", "B"): "P7, O1",
    (13, "19-06-26", "B"): "P8",
    (10, "26-06-26", "B"): "P8",
    (10, "24-06-26", "B"): "P7",
    (10, "22-06-26", "B"): "Fp2,P8",
    (10, "19-06-26", "B"): "P8",
    (8, "19-06-26", "B"): "P8, P7, AF8, AF7",
    (6, "25-06-26", "B"): "O2",
    (6, "19-06-26", "B"): "P8",
    (2, "22-06-26", "B"): "O1,AF8"
}

BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (13, 30),
    "Gamma": (30, 40)
}

def get_total_fif_files():
    fif_dict = {}
    for f in mne_files_path.rglob("*.fif"):
        fif_dict[f.name] = str(f)
    #print(f"Found {len(fif_dict)} FIF files")
    return fif_dict
fif_dict = get_total_fif_files()

def get_fif_path(row):
    expected_name = (f"ID{row['ID']}_{row['DATE']}_{row['LABEL']}_raw.fif")
    return fif_dict.get(expected_name, None)

def get_partipants(): 
    return [2,5,6,8,10,13]

def add_bad_channels_column(info_df):
    if "BAD CHANNELS" not in info_df.columns:
        info_df["BAD CHANNELS"] = ""
    for idx, row in info_df.iterrows():
        key = (
            row["ID"],
            row["DATE"],
            row["LABEL"]
        )

        if key in BAD_CHANNELS:
            info_df.loc[idx, "BAD CHANNELS"] = BAD_CHANNELS[key]

    return info_df

def load_dataset():
    info_df = pd.read_csv(info_datasets).reset_index(drop=True)
    if "BAD CHANNELS" in info_df.columns:
        info_df["BAD CHANNELS"] = (info_df["BAD CHANNELS"].fillna("").astype(str))
    if "ICA PATH" in info_df.columns:
        info_df["ICA PATH"] = (info_df["ICA PATH"].fillna("").astype(str))
    if "ICA EXCLUDE" in info_df.columns:
        info_df["ICA EXCLUDE"] = (info_df["ICA EXCLUDE"].fillna("").astype(str))

    info_df["PSYCHOPY"] = (info_df["PSYCHOPY"].str.replace("/content/drive/My Drive/", r"G:/Mon Drive/", regex=False))
    info_df["TRIMMED RAW EEG"] = (info_df["TRIMMED RAW EEG"].str.replace( "/content/drive/My Drive/",r"G:/Mon Drive/",regex=False))
    info_df["RAW EEG"] = (info_df["RAW EEG"].str.replace("/content/drive/My Drive/",r"G:/Mon Drive/",regex=False))
    #info_df.drop["RAW EEG, TRIMMED RAW EEG"]
    info_df["MNE PATH"] = info_df.apply(get_fif_path,axis=1)
    return info_df

def clean_setup_signal(df, row):

    mne_path = df.loc[row, "MNE PATH"]
    date = df.loc[row, "DATE"]
    id = df.loc[row, "ID"]
    phase = df.loc[row, "LABEL"]

    raw = mne.io.read_raw_fif(mne_path, preload=True)
    print("Channels:", raw.ch_names)
    annotations = get_markers_annotations(id=id,date=date,phase=phase)
    raw.set_annotations(annotations)
    
    #Bad channels 
    if "BAD CHANNELS" in df.columns:
        bads = df.loc[row, "BAD CHANNELS"]
        if (pd.notna(bads) and bads != "" and bads.lower() != "nan"):
            raw.info["bads"] = [ch.strip() for ch in bads.split(",")]

    print("Bad channels:", raw.info["bads"])
    
    # BandPass and Notch Filtering
    raw.filter(l_freq=1,h_freq=40)
    raw.notch_filter(freqs=50)

    # Average reference
    if len(raw.info["bads"]) < len(raw.ch_names):
        raw.set_eeg_reference("average")
    else:
        print("All channels marked as bad.")
        return None
    
    raw.plot()
    
    return raw

def get_psd(mne_object, row, df, pre_phase = False, post_phase = False, overall = False): 
    spectrum = mne_object.compute_psd()
    
    if pre_phase: 
        fig = spectrum.plot(average=False, exclude="bads", amplitude=False)
        fig.canvas.manager.set_window_title(f"PSD PRE EEG - Eyes Open | ID{df.loc[row,'ID']} | {df.loc[row,'DATE']} | {df.loc[row,'LABEL']}")
    elif post_phase: 
        fig = spectrum.plot(average=False, exclude="bads", amplitude=False)
        fig.canvas.manager.set_window_title(f"PSD POST EEG - Eyes Open | ID{df.loc[row,'ID']} | {df.loc[row,'DATE']} | {df.loc[row,'LABEL']}")
    elif overall: 
        fig = spectrum.plot(average=True, exclude="bads", amplitude=False)
        fig.canvas.manager.set_window_title(f"PSD EEG - Eyes Open | ID{df.loc[row,'ID']} | {df.loc[row,'DATE']} | {df.loc[row,'LABEL']}")

    plt.show()
    if pre_phase: 
        fig.savefig(Path(results_EEG_path,f"ID{df.loc[row,'ID']}", f"PSD PRE EEG - Eyes Open_ID{df.loc[row,'ID']}_{df.loc[row,'DATE']}_{df.loc[row,'LABEL']}.png"), dpi=300, bbox_inches="tight")

    elif post_phase: 
        fig.savefig(Path(results_EEG_path,f"ID{df.loc[row,'ID']}", f"PSD POST EEG - Eyes Open_ID{df.loc[row,'ID']}_{df.loc[row,'DATE']}_{df.loc[row,'LABEL']}.png"), dpi=300, bbox_inches="tight")
    

def get_sensors_plot(mne_object):
    mne_object.plot_sensors(show_names=True)

def look_for_artifacts(raw):
    print("Bad channels:", raw.info["bads"])
    ica = mne.preprocessing.ICA(
        n_components=0.99,
        random_state=97,
        max_iter="auto"
    )
    ica.fit(raw, reject_by_annotation=True)
    print(f"ICA components: {ica.n_components_}")
    #eog_idx, scores = ica.find_bads_eog(raw)
    #ica.plot_scores(scores, exclude=eog_idx)
    #print(f"EOG INDEX: {eog_idx}, SCORES: {scores}")
    ica.plot_components()
    plt.show(block = False)
    ica.plot_sources(raw)
    plt.show(block = False)
    print("-----------------------------------------------------------------------------")
    labels = label_components(raw, ica, method="iclabel")
    for i, (label, probs) in enumerate(zip(labels["labels"], labels["y_pred_proba"])):
        confidence = np.max(probs)
        print(f"IC{i}: {label} ({confidence:.2f})")

    components_to_inspect = input("Components to inspect (e.g. 0,1): ").split(",")
    components_to_inspect = [int(x.strip())for x in components_to_inspect if x.strip() != ""]

    if len(components_to_inspect) > 0:
        ica.plot_properties(raw, picks=components_to_inspect)

    plt.show(block = False)


    ica_to_exclude = input("ICA COMPONENTS TO BE REMOVED(e.g. [1,2]): ").split(",")
    ica_to_exclude = [int(x.strip()) for x in ica_to_exclude if x.strip() != ""]
    ica.exclude = ica_to_exclude
    print(f"Components excluded: {ica.exclude}")
    return ica


def first_phase(df,row):
    date = df.loc[row,"DATE"] 
    id = df.loc[row, "ID"]
    phase = df.loc[row, "LABEL"]
    print(f"Date {date} - ID: {id} - Phase {phase}")
    raw = clean_setup_signal(df, row)
    if raw is None:
        print("Skipping session because all channels are bad.")
        return None
    ica = look_for_artifacts(raw)
    raw_clean = raw.copy()
    ica.apply(raw_clean)
    raw.plot(block=False, title=f"ID{id} - {date} - {phase} Before ICA")
    plt.show(block = False)
    raw_clean.plot(block=True, title=f"ID{id} - {date} - {phase}")
    #plt.show()

    #--------------------------------------------------------------------------------
    ica_path = save_mne(date,phase,id,data_mne_clean=raw_clean)
    
    if "ICA PATH" not in df.columns:
        df["ICA PATH"] = ""
    if "ICA EXCLUDE" not in df.columns:
        df["ICA EXCLUDE"] = ""
    df.loc[row, "ICA PATH"] = ica_path
    if ica is not None:
        df.loc[row, "ICA EXCLUDE"] = ",".join(map(str, ica.exclude))
    else:
        df.loc[row, "ICA EXCLUDE"] = ""

    #df.to_csv(info_datasets, index=False)
    update_session(df, row)
    raw_clean.plot(block=False,scalings="auto",title=f"ID{id} - {date} - {phase}")
    ica_path = save_mne(date,phase,id,data_mne_clean=raw_clean) # in case I want to change something 
    #get_psd(mne_object=raw_clean)
    get_psd(mne_object = raw_clean, row = row, df =df, pre_phase = False, post_phase = False, overall = True)
    #---------------------------------------------------------------------------------
    return raw_clean

    
def get_markers_annotations(id, date,phase):
    expected_name = f"ID{id}_{date}_{phase}_markers.csv"
    for f in Path(mne_files_path).rglob(f"*markers.csv"):
        if f.name == expected_name: 
        #if (f"ID{id}_" in f.name and f"_{date}_" in f.name and f"_{phase}_" in f.name):
            markers = pd.read_csv(f)
            print(f)
            print("------------------")
            print(markers.head())
            print(Counter(markers["marker"]))
            idx_1000 = markers.loc[markers["marker"] == 90, "onset"]
            print(f"idx_1000:{idx_1000}")
            idx_1003 = markers.loc[markers["marker"] == 1003, "onset"]
            print(f"idx_1003:{idx_1003}")
            idx_90 = markers.loc[markers["marker"] == 90, "onset"]
            idx_91 = markers.loc[markers["marker"] == 91, "onset"]
            idx_92 = markers.loc[markers["marker"] == 92, "onset"]
            idx_93 = markers.loc[markers["marker"] == 93, "onset"]
            print(f"idx_90:{idx_90}, idx_91:{idx_91},idx_92:{idx_92}, idx_93:{idx_93}")

            idx_1 = markers.loc[markers["marker"] == 1, "onset"]
            idx_2 = markers.loc[markers["marker"] == 2, "onset"]
            print(f"idx_1:{idx_1}, idx_2: {idx_2}")
            raw_markers = markers.loc[(markers["marker"] == 1)  | (markers["marker"] == 2)]

            print(raw_markers.head(20))


            annotations = mne.Annotations(
            onset=markers["onset"].to_numpy(),
            duration=np.zeros(len(markers)),
            description=markers["marker"].astype(str).to_numpy())
            return annotations
        else:
            continue

def save_mne(date, phase, id, data_mne_clean):
    save_path = Path(mne_files_path,f"ID{id}",f"ID{id}_{date}_{phase}_ica_raw.fif")
    data_mne_clean.save(save_path,overwrite=True)
    print(f"Save: {save_path}")
    return str(save_path)

def preprocess_all_sessions():
    info_df = load_dataset()
    info_df = add_bad_channels_column(info_df)

    for row in info_df.index:
        print(info_df.loc[row, ["ID","DATE","LABEL"]])
        if ("ICA PATH" in info_df.columns and pd.notna(info_df.loc[row, "ICA PATH"]) and info_df.loc[row, "ICA PATH"] != ""):
            print(f"Already processed ID {info_df.loc[row, 'ID']}. Skipping.")
            continue
        first_phase(info_df, row)

def first_phase_example(df,row):
    date = df.loc[row,"DATE"] 
    id = df.loc[row, "ID"]
    phase = df.loc[row, "LABEL"]
    raw = clean_setup_signal(df, row)
    if raw is None:
        print("Skipping session because all channels are bad.")
        return None
    ica = look_for_artifacts(raw)
    raw_clean = raw.copy()
    ica.apply(raw_clean)
    raw.plot()
    plt.show(block = False)
    raw_clean.plot()
    plt.show()
    return raw_clean

def example():
    df = load_dataset()
    df = add_bad_channels_column(df)
    row = 0
    raw_clean = first_phase_example(df,row)
    if raw_clean is not None:
        print(raw_clean.ch_names)

def inspect_annotations_clean_data(row):
    df = load_dataset()
    try: 
        ica_path = df.loc[row,"ICA PATH"]
        mne_object = mne.io.read_raw_fif(ica_path, preload = True)
        print(mne_object.annotations)
        annotations_df = pd.DataFrame({
        "onset": mne_object.annotations.onset,
        "duration": mne_object.annotations.duration,
        "description": mne_object.annotations.description})
        return annotations_df,mne_object
    except IndexError: 
        print(f"ID {df.loc[row, "ID"]} has not been preprocessed.")
        return None 

def crop_pre_post_phase(row): 
    annotations_df,mne_object = inspect_annotations_clean_data(row)
    if annotations_df is None: 
        return None
    else: 
        onset_start_pre = annotations_df[annotations_df["description"] == "90"]["onset"].iloc[0] # But I would like to start with that -2 seconds delay 
        onset_start_post = annotations_df[annotations_df["description"] == "92"]["onset"].iloc[0]-2 # -2 seconds
        end_ding_pre = annotations_df[annotations_df["description"] == "1003"].iloc[0]["onset"] + 2 # windows
        end_ding_post = annotations_df[annotations_df["description"] == "1003"].iloc[2]["onset"] + 2 # windows 
        pre_phase_mne_object = mne_object.copy().crop(tmin = onset_start_pre, tmax = end_ding_pre+2)
        post_phase_mne_object = mne_object.copy().crop(tmin = onset_start_post, tmax = end_ding_post+2)
        return pre_phase_mne_object, post_phase_mne_object

def spatial_frequency_information(df,row,pre_phase_mne_object, post_phase_mne_object):
    fig = pre_phase_mne_object.compute_psd().plot_topomap(show = False)
    fig.canvas.manager.set_window_title(f"PRE EEG - Eyes Open | ID{df.loc[row,'ID']} | {df.loc[row,'DATE']} | {df.loc[row,'LABEL']}")
    plt.show(block = False)
    fig.savefig(Path(results_EEG_path,f"ID{df.loc[row,'ID']}", f"PRE EEG - Eyes Open_ID{df.loc[row,'ID']}_{df.loc[row,'DATE']}_{df.loc[row,'LABEL']}.png"), dpi=300, bbox_inches="tight")

    fig = post_phase_mne_object.compute_psd().plot_topomap(show = False)
    fig.canvas.manager.set_window_title(f"POST EEG - Eyes Open | ID{df.loc[row,'ID']} | {df.loc[row,'DATE']} | {df.loc[row,'LABEL']}")
    plt.show()
    fig.savefig(Path(results_EEG_path,f"ID{df.loc[row,'ID']}", f"POST EEG - Eyes Open_ID{df.loc[row,'ID']}_{df.loc[row,'DATE']}_{df.loc[row,'LABEL']}.png"), dpi=300, bbox_inches="tight")
def make_epochs_from_phase(raw_phase, duration=2.0, overlap=1.0):

    epochs = mne.make_fixed_length_epochs(
        raw_phase,
        duration=duration,
        overlap=overlap,
        preload=True,
        reject_by_annotation=True
    )

    return epochs

def plot_morlet_tfr(df, row):
    bad_channels = df.loc[row,"BAD CHANNELS"]
    if pd.isna(bad_channels) or bad_channels == "" or bad_channels.lower() == "nan":
        bad_channels = []
    else:
        bad_channels = [ch.strip() for ch in bad_channels.split(",")]

    pre_phase, post_phase = crop_pre_post_phase(row)

    pre_epochs = make_epochs_from_phase(pre_phase)
    post_epochs = make_epochs_from_phase(post_phase)

    freqs = np.arange(1, 45, 1)
    n_cycles = freqs / 2

    pre_power = pre_epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        average=True,
        return_itc=False
    )

    post_power = post_epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        average=True,
        return_itc=False
    )

    
    occipital = [ch for ch in ["O1", "O2", "P7", "P8"] if ch not in bad_channels]

    frontal = [ch for ch in ["AF7", "Fp1", "Fp2", "AF8"] if ch not in bad_channels]

    if len(occipital) !=0: 
        fig = pre_power.plot(
            picks=occipital,
            combine="mean",
            title=f"PRE Occipital | ID{df.loc[row, 'ID']}"
        )
        #fig = plt.gcf()
        #print(type(fig))
        plt.show(block = False)
        fig[0].savefig(Path(results_EEG_path,f"ID{df.loc[row,'ID']}", f"PRE OCCIPITAL WAVELET - Eyes Open_ID{df.loc[row,'ID']}_{df.loc[row,'DATE']}_{df.loc[row,'LABEL']}.png"), dpi=300, bbox_inches="tight")

    if len(occipital) !=0: 
        fig = post_power.plot(
            picks=occipital,
            combine="mean",
            title=f"POST Occipital | ID{df.loc[row, 'ID']}"
        )
        #fig = plt.gcf()
        plt.show(block = False)
        fig[0].savefig(Path(results_EEG_path,f"ID{df.loc[row,'ID']}", f"POST OCCIPITAL WAVELET - Eyes Open_ID{df.loc[row,'ID']}_{df.loc[row,'DATE']}_{df.loc[row,'LABEL']}.png"), dpi=300, bbox_inches="tight")

    if len(frontal) !=0: 
        fig = pre_power.plot(
            picks=frontal,
            combine="mean",
            title=f"PRE Frontal | ID{df.loc[row, 'ID']}"
        )
        #fig = plt.gcf()
        plt.show(block = False)
        fig[0].savefig(Path(results_EEG_path,f"ID{df.loc[row,'ID']}", f"PRE FRONTAL WAVELET - Eyes Open_ID{df.loc[row,'ID']}_{df.loc[row,'DATE']}_{df.loc[row,'LABEL']}.png"), dpi=300, bbox_inches="tight")

    if len(frontal) !=0: 
        fig = post_power.plot(
            picks=frontal,
            combine="mean",
            title=f"POST Frontal | ID{df.loc[row, 'ID']}"
        )
        #fig = plt.gcf()
        plt.show()
        fig[0].savefig(Path(results_EEG_path,f"ID{df.loc[row,'ID']}", f"POST FRONTAL WAVELET - Eyes Open_ID{df.loc[row,'ID']}_{df.loc[row,'DATE']}_{df.loc[row,'LABEL']}.png"), dpi=300, bbox_inches="tight")


def plot_time_frequency_analysis(df, row):
    pre_phase_mne_object, post_phase_mne_object = crop_pre_post_phase(row)
    spatial_frequency_information(df,row,pre_phase_mne_object, post_phase_mne_object)
    plot_morlet_tfr(df, row)

    get_psd(mne_object = pre_phase_mne_object, row = row, df = df, pre_phase = True)
    get_psd(mne_object = post_phase_mne_object, row = row, df = df, post_phase = True)

    

    # f"Wavelet | ID{id} | {date} | {phase} | {label} PHASE"
    #fig.canvas.manager.set_window_title(f"ID{df.loc[row,'ID']} PRE EEG - Eyes Open")
def get_region_psd(raw, channels):

    picks = mne.pick_channels(raw.ch_names,include=channels)

    spectrum = raw.compute_psd(
        method="welch",
        picks=picks,
        fmin=0.5,
        fmax=40
    )

    psds = spectrum.get_data()
    freqs = spectrum.freqs
    region_psd = psds.mean(axis=0)
    print(psds.shape)

    return region_psd, freqs

def compute_absolute_power(psd, freqs):

    results = {}

    for band,(fmin,fmax) in BANDS.items():

        idx = (freqs >= fmin) & (freqs < fmax)

        results[band] = np.trapezoid(
            psd[idx],
            freqs[idx]
        )

    return results

def compute_relative_power(abs_power):
    total_power = sum(abs_power.values())
    rel_power = {}
    for band,power in abs_power.items():
        rel_power[f"{band}_rel"] = (power / total_power)
    return rel_power

def compute_ratios(abs_power):

    beta = abs_power["Beta"]

    if beta == 0:
        ratio = np.nan
    else:
        ratio = abs_power["Theta"] / beta

    return {
        "Theta_Beta": ratio
    }

def main(df): 
     for row in range(9):
        plot_time_frequency_analysis(df, row)
        print(f"ID {df.loc[row, "ID"]} done")

def extract_band_features(raw, channels):

    psd,freqs = get_region_psd(raw,channels)

    abs_power = compute_absolute_power(psd, freqs)

    rel_power = compute_relative_power(abs_power)

    ratios = compute_ratios(abs_power)

    return {
        **abs_power,
        **rel_power,
        **ratios
    }

def create_dataframe_values(df):

    rows = []

    for row in df.index:

        print(
            f"Processing ID {df.loc[row,'ID']} "
            f"{df.loc[row,'DATE']} "
            f"{df.loc[row,'LABEL']}"
        )

        try:
            pre_phase, post_phase = crop_pre_post_phase(row)
            bad_channels = pre_phase.info["bads"]

            occipital = [
                ch for ch in ["O1","O2","P7","P8"]
                if ch not in bad_channels
            ]

            frontal = [
                ch for ch in ["AF7","Fp1","Fp2","AF8"]
                if ch not in bad_channels
            ]


            results = {
                "ID": df.loc[row,"ID"],
                "DATE": df.loc[row,"DATE"],
                "PHASE": df.loc[row,"LABEL"],
                "OCC_CHANNELS": ",".join(occipital),
                "FRONT_CHANNELS": ",".join(frontal)
            }

            # ---------------- PRE ----------------

            if len(occipital) > 0:

                pre_occ = extract_band_features(
                    pre_phase,
                    occipital
                )

                results.update({
                    f"PRE_OCC_{k}": v
                    for k,v in pre_occ.items()
                })

            if len(frontal) > 0:

                pre_front = extract_band_features(
                    pre_phase,
                    frontal
                )

                results.update({
                    f"PRE_FRONT_{k}": v
                    for k,v in pre_front.items()
                })

            # ---------------- POST ----------------

            if len(occipital) > 0:

                post_occ = extract_band_features(
                    post_phase,
                    occipital
                )

                results.update({
                    f"POST_OCC_{k}": v
                    for k,v in post_occ.items()
                })

            if len(frontal) > 0:

                post_front = extract_band_features(
                    post_phase,
                    frontal
                )

                results.update({
                    f"POST_FRONT_{k}": v
                    for k,v in post_front.items()
                })

            rows.append(results)

        except Exception as e:

            print(
                f"Error in row {row}: {e}"
            )

    return pd.DataFrame(rows)
def extract_epoch_band_features(
    raw,
    channels,
    duration=2.0,
    overlap=1.0
):

    epochs = make_epochs_from_phase(
        raw,
        duration=duration,
        overlap=overlap
    )

    rows = []

    for epoch_idx in range(len(epochs)):

        epoch = epochs[epoch_idx]

        spectrum = epoch.compute_psd(
            method="welch",
            picks=channels,
            fmin=0.5,
            fmax=40
        )

        psds = spectrum.get_data()
        freqs = spectrum.freqs

        psd = psds.mean(axis=0).mean(axis=0)
        if epoch_idx == 0:
            print("PSD shape:", psds.shape)

        abs_power = compute_absolute_power(
            psd,
            freqs
        )

        rel_power = compute_relative_power(
            abs_power
        )

        ratios = compute_ratios(
            abs_power
        )

        rows.append({
            "EPOCH": epoch_idx,
            **abs_power,
            **rel_power,
            **ratios
        })

    return pd.DataFrame(rows)
def extract_band_features_epochs(
    raw,
    channels,
    duration=2,
    overlap=1
):

    epochs = make_epochs_from_phase(
        raw,
        duration=duration,
        overlap=overlap
    )

    results = {
        "Delta": [],
        "Theta": [],
        "Alpha": [],
        "Beta": [],
        "Gamma": [],
        "Theta_Beta": []
    }

    for epoch in epochs:

        psd, freqs = get_region_psd(
            epoch,
            channels
        )

        abs_power = compute_absolute_power(
            psd,
            freqs
        )

        results["Delta"].append(
            abs_power["Delta"]
        )

        results["Theta"].append(
            abs_power["Theta"]
        )

        results["Alpha"].append(
            abs_power["Alpha"]
        )

        results["Beta"].append(
            abs_power["Beta"]
        )

        results["Gamma"].append(
            abs_power["Gamma"]
        )

        results["Theta_Beta"].append(
            abs_power["Theta"] /
            abs_power["Beta"]
        )

    return results

def create_epoch_dataframe(df):

    rows = []

    for row in df.index:

        print(
            f"Processing ID {df.loc[row,'ID']} "
            f"{df.loc[row,'DATE']} "
            f"{df.loc[row,'LABEL']}"
        )

        try:

            pre_phase, post_phase = crop_pre_post_phase(row)

            bad_channels = pre_phase.info["bads"]

            occipital = [
                ch for ch in ["O1","O2","P7","P8"]
                if ch not in bad_channels
            ]

            frontal = [
                ch for ch in ["AF7","Fp1","Fp2","AF8"]
                if ch not in bad_channels
            ]

            for period_name, raw_period in [
                ("PRE", pre_phase),
                ("POST", post_phase)
            ]:

                for region_name, channels in [
                    ("OCC", occipital),
                    ("FRONT", frontal)
                ]:

                    if len(channels) == 0:
                        continue

                    epoch_df = extract_epoch_band_features(
                        raw_period,
                        channels,
                        duration=2.0,
                        overlap=1.0
                    )

                    epoch_df["ID"] = df.loc[row,"ID"]
                    epoch_df["DATE"] = df.loc[row,"DATE"]
                    epoch_df["PHASE"] = df.loc[row,"LABEL"]

                    epoch_df["PERIOD"] = period_name
                    epoch_df["REGION"] = region_name
                    epoch_df["CHANNELS"] = ",".join(channels)

                    rows.append(epoch_df)

        except Exception as e:

            print(
                f"Error row {row}: {e}"
            )

    return pd.concat(
        rows,
        ignore_index=True
    )

def create_epoch_feature_dataframe():

    df = load_dataset()

    epoch_df = create_epoch_dataframe(df)

    print(epoch_df.head())

    epoch_df.to_csv(
        Path(
            results_EEG_path,
            "eeg_epoch_dataframe.csv"
        ),
        index=False
    )

def create_dataframe(): 
    df = load_dataset()
    feature_df = create_dataframe_values(df)
    print(feature_df)
    feature_df.to_csv(Path(results_EEG_path, "eeg_dataframe.csv"), index=False)

def update_session(df, row):

    old_df = pd.read_csv(info_datasets)

    keys = ["ID", "DATE", "LABEL"]

    mask = (
        (old_df["ID"] == df.loc[row, "ID"]) &
        (old_df["DATE"] == df.loc[row, "DATE"]) &
        (old_df["LABEL"] == df.loc[row, "LABEL"])
    )

    old_df.loc[mask, "ICA PATH"] = df.loc[row, "ICA PATH"]
    old_df.loc[mask, "ICA EXCLUDE"] = df.loc[row, "ICA EXCLUDE"]
    old_df.loc[mask, "BAD CHANNELS"] = df.loc[row, "BAD CHANNELS"]
    old_df.loc[mask, "MNE PATH"] = df.loc[row, "MNE PATH"]

    old_df.to_csv(info_datasets, index=False)

def preprocess_pipeline(): 
    df = load_dataset()
    df = add_bad_channels_column(df)
    row = 61
    #raw_test = mne.io.read_raw_fif(df.loc[row, "ICA PATH"], preload=False)
    #print(raw_test.annotations)
    raw_clean = first_phase(df,row)
    if raw_clean is not None:
        print(raw_clean.ch_names)
    plot_time_frequency_analysis(df, row)


def add_epoch_behavior(row, epochs): 
    df = load_dataset()
    behavior_df = pd.read_csv(final_dataframe_behavior_per_trials)
    behavior_df["date"] = pd.to_datetime(behavior_df["date"])

    id = df.loc[row, "ID"]
    date = pd.to_datetime(
    df.loc[row, "DATE"],
    format="%d-%m-%y")
    phase = df.loc[row, "LABEL"]

    behavior = behavior_df[
    (behavior_df["ID"] == id) &
    (pd.to_datetime(behavior_df["date"]) == date) &
    (behavior_df["phase"] == phase)].copy()
    behavior = behavior.sort_values("trial")
    behavior = behavior.reset_index(drop=True)
    print(f"Epochs: {len(epochs)}")
    print(f"Behavior: {len(behavior)}")

    assert len(epochs) == len(behavior), (
    f"Mismatch! Epochs={len(epochs)} "
    f"Behavior={len(behavior)} "
    f"ID={id} DATE={date} PHASE={phase}")
    print(behavior[["trial", "stimulus_type"]].head())
    print(behavior.columns)
    epochs.metadata = behavior 
    return epochs


def plot_epoch_go_no_go_per_all_id():

    df = load_dataset()
    rejected_sessions = []

    for row in range(df.shape[0]):

        id = df.loc[row, "ID"]
        date = df.loc[row, "DATE"]
        phase = df.loc[row, "LABEL"]

        epochs = create_epoch_go_no_go(row=row)
        epochs = add_epoch_behavior(row=row, epochs=epochs)

        n_before = len(epochs)

        epochs = reject_bad_erp_epochs(epochs=epochs)

        if len(epochs) >= 0.5 * n_before:

            plot_avg_epochs_per_channels(
                epochs=epochs,
                row=row,
                df=df
            )

            plot_epochs_per_region(
                epochs=epochs,
                row=row,
                df=df
            )

            epochs.save(
                Path(
                    mne_root_epochs,
                    f"ID{id}",
                    f"ID{id}_{date}_{phase}-epo.fif"
                ),
                overwrite=True
            )

        else:

            rejected_sessions.append({
                "ID": id,
                "DATE": date,
                "PHASE": phase,
                "Remaining": len(epochs),
                "Original": n_before
            })

            print(
                f"ID{id}_{date}_{phase} didn't pass "
                f"({len(epochs)}/{n_before} epochs)"
            )

    print("\n" + "="*60)
    print("Rejected sessions")
    print("="*60)

    for s in rejected_sessions:

        print(
            f"ID{s['ID']} | "
            f"{s['DATE']} | "
            f"{s['PHASE']} | "
            f"{s['Remaining']}/{s['Original']} epochs"
        )


def create_epoch_go_no_go(row): 
    df = load_dataset()
    mne_object = mne.io.read_raw_fif(df.loc[row, "ICA PATH"], preload=True)
    #mne_object.plot(block = True)
    events, event_id = mne.events_from_annotations(
    mne_object,
    event_id={
        "1": 1,      # No-Go
        "2": 2     # Go
    })
    print(events[:5, 2])
    

    n_go = np.sum(events[:, 2] == 2)
    n_nogo = np.sum(events[:, 2] == 1)
    print(f"Go:     {n_go}")
    print(f"No-Go:  {n_nogo}")

    epochs = mne.Epochs(
    mne_object,
    events,
    event_id={
        "NoGo":1,
        "Go":2
    },
    tmin=-0.2,
    tmax=0.8,
    baseline=(-0.2,0),
    preload=True,
    reject_by_annotation=True,
    event_repeated="drop")

    print(epochs)
    print(f"Total epochs: {len(epochs)}")
    #epochs["Go"].plot_image()
    #epochs["NoGo"].plot_image()
    #epochs["Go"].average().plot()
    #epochs["NoGo"].average().plot()
    #------------------------------------------
    epochs = epochs.filter(
    l_freq=None,
    h_freq=18)
    return epochs 

def plot_avg_epochs_per_channels(epochs, row, df): 

    fig, axs = plt.subplots(
    2,
    4,
    figsize=(16,8),
    sharex=True,
    sharey=False)
    bad_channels = epochs.info["bads"]

    if bad_channels is None:
        bad_channels = []

    evokeds = {
        "Go": epochs["Go"].average(),
        "NoGo": epochs["NoGo"].average()
    }

    channels = ["Fp1","Fp2","AF8","AF7",
            "P7","P8","O1","O2"]

    for ch, ax in zip(channels, axs.ravel()):

        if ch in bad_channels:
            ax.set_title(f"{ch}\nBad channel")
            ax.axis("off")
            continue
        print("="*50)
        print(f"ID: {df.loc[row,'ID']}")
        print(f"DATE: {df.loc[row,'DATE']}")
        print(f"CHANNEL: {ch}")
        print("Channels available:", epochs.ch_names)
        mne.viz.plot_compare_evokeds(
            evokeds,
            picks=ch,
            axes=ax,
            show=False,
            legend=False
        )
        ax.set_title(ch)

    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    #plt.show()
    fig.savefig(Path(results_EEG_path,f"ID{df.loc[row,'ID']}", f"ERP_all channels_ID{df.loc[row,'ID']}_{df.loc[row,'DATE']}_{df.loc[row,'LABEL']}.png"), dpi=300, bbox_inches="tight")



def roi_wave(evoked, channels):
    if evoked is None:
        return None
    if len(channels) == 0:
        return None

    return (
        evoked.copy()
        .pick(channels)
        .data
        .mean(axis=0)
    )


def bootstrap_roi(epochs_condition, channels, B=1000):
    if len(epochs_condition) == 0:
        return None,None,None
    if len(channels) == 0:
        return None,None,None

    data = (
        epochs_condition.copy()
        .pick(channels)
        .get_data()
    )

    n_epochs = data.shape[0]
    boot = []

    for _ in range(B):
        idx = np.random.choice(
            n_epochs,
            size=n_epochs,
            replace=True
        )

        wave = (
            data[idx]
            .mean(axis=0)
            .mean(axis=0)
        )

        boot.append(wave)

    boot = np.array(boot)

    mean = boot.mean(axis=0)
    ci_low = np.percentile(boot, 2.5, axis=0)
    ci_high = np.percentile(boot, 97.5, axis=0)

    return mean,ci_low,ci_high


def plot_bootstrap(ax, times, mean, ci_low, ci_high, label, color):
    if mean is None:
        return

    ax.plot(
        times,
        mean,
        label=label,
        linewidth=2, 
        color = color 
    )

    ax.fill_between(
        times,
        ci_low,
        ci_high,
        alpha=0.10,
        color = color 
    )


def safe_average(epochs_condition):
    if len(epochs_condition) == 0:
        return None

    return epochs_condition.average()


def detect_peak(
    wave,
    times,
    tmin,
    tmax,
    polarity
):

    if wave is None:
        return None

    mask = (
        (times >= tmin)
        &
        (times <= tmax)
    )

    wave_window = wave[mask]
    time_window = times[mask]

    if len(wave_window) == 0:
        return None

    if polarity == "positive":

        peaks, properties = find_peaks(
            wave_window,
            prominence=0
        )

    elif polarity == "negative":

        peaks, properties = find_peaks(
            -wave_window,
            prominence=0
        )

    else:
        raise ValueError(
            "polarity must be 'positive' or 'negative'"
        )

    if len(peaks) == 0:
        return None

    prominences = properties["prominences"]

    best_peak = peaks[
        np.argmax(prominences)
    ]

    return {
        "latency": time_window[best_peak],
        "amplitude": wave_window[best_peak],
        "index": best_peak
    }

def latency_amplitude_peaks(
    roi_name,
    correct_go,
    correct_nogo,
    omission,
    commission,
    times
):

    ERP_COMPONENTS = {

        "P2": ((0.10,0.25),"positive"),
        "N2": ((0.22,0.45),"negative"),
        "P3": ((0.30,0.70),"positive")

    }

    CONDITIONS = {

        "Go": correct_go,
        "NoGo": correct_nogo,
        "Omission": omission,
        "Commission": commission

    }

    peaks = {}
    features = {}

    for condition_name, wave in CONDITIONS.items():

        for component_name, (window, polarity) in ERP_COMPONENTS.items():

            peak = detect_peak(
                wave,
                times,
                *window,
                polarity=polarity
            )

            peaks[
                f"{roi_name}_{component_name}_{condition_name}"
            ] = peak

            if peak is None:

                features[
                    f"{roi_name}_{component_name}_{condition_name}_Amp"
                ] = np.nan

                features[
                    f"{roi_name}_{component_name}_{condition_name}_Lat"
                ] = np.nan

            else:

                features[
                    f"{roi_name}_{component_name}_{condition_name}_Amp"
                ] = peak["amplitude"]

                features[
                    f"{roi_name}_{component_name}_{condition_name}_Lat"
                ] = peak["latency"]

    return peaks, features

def plot_peak(ax, peak, label, color):

    if peak is None:
        return

    ax.scatter(
        peak["latency"],
        peak["amplitude"],
        marker="v",
        s=70,
        color=color,
        edgecolor="black",
        zorder=100
    )

    ax.text(
        peak["latency"],
        peak["amplitude"],
        label,
        color=color,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

def plot_epochs_per_region(epochs, row, df):

    MIN_OMISSION_COMISSION = 30

    id = df.loc[row,"ID"]
    date = df.loc[row,"DATE"]
    phase = df.loc[row,"LABEL"]

    #FRONTAL = ["AF7", "Fp1", "Fp2", "AF8"]
    FRONTAL = ["Fp1", "Fp2"]
    OCCIPITAL = ["P7", "P8", "O1", "O2"]

    bad = epochs.info["bads"]

    frontal = [
        ch for ch in FRONTAL
        if ch not in bad
    ]

    occipital = [
        ch for ch in OCCIPITAL
        if ch not in bad
    ]

    correct_go = epochs[
        "(stimulus_type == 'Go') and (correct == 'Correct')"
    ]

    correct_nogo = epochs[
        "(stimulus_type == 'No_Go') and (correct == 'Correct')"
    ]

    omission = epochs[
        "(stimulus_type == 'Go') and (correct == 'Incorrect')"
    ]

    commission = epochs[
        "(stimulus_type == 'No_Go') and (correct == 'Incorrect')"
    ]

    plot_omission = len(omission) >= MIN_OMISSION_COMISSION
    plot_commission = len(commission) >= MIN_OMISSION_COMISSION

    print(f"Correct Go:    {len(correct_go)}")
    print(f"Correct NoGo:  {len(correct_nogo)}")
    print(f"Omission:      {len(omission)}")
    print(f"Commission:    {len(commission)}")
    print(f"Frontal channels: {frontal}")
    print(f"Occipital channels: {occipital}")

    if not plot_omission:
        print(f"Omission not plotted: only {len(omission)} trials")

    if not plot_commission:
        print(f"Commission not plotted: only {len(commission)} trials")

    correct_go_evoked = safe_average(correct_go)
    correct_nogo_evoked = safe_average(correct_nogo)

    if (
        correct_go_evoked is None
        and correct_nogo_evoked is None
    ):
        print("No epochs available.")
        return

    times = (
        correct_go_evoked
        or
        correct_nogo_evoked
    ).times

    correct_go_front, correct_go_front_low, correct_go_front_high = bootstrap_roi(
        correct_go,
        frontal
    )

    correct_go_occ, correct_go_occ_low, correct_go_occ_high = bootstrap_roi(
        correct_go,
        occipital
    )

    correct_nogo_front, correct_nogo_front_low, correct_nogo_front_high = bootstrap_roi(
        correct_nogo,
        frontal
    )

    correct_nogo_occ, correct_nogo_occ_low, correct_nogo_occ_high = bootstrap_roi(
        correct_nogo,
        occipital
    )

    omission_front = omission_front_low = omission_front_high = None
    omission_occ = omission_occ_low = omission_occ_high = None

    commission_front = commission_front_low = commission_front_high = None
    commission_occ = commission_occ_low = commission_occ_high = None

    if plot_omission:

        omission_front, omission_front_low, omission_front_high = bootstrap_roi(
            omission,
            frontal
        )

        omission_occ, omission_occ_low, omission_occ_high = bootstrap_roi(
            omission,
            occipital
        )

    if plot_commission:

        commission_front, commission_front_low, commission_front_high = bootstrap_roi(
            commission,
            frontal
        )

        commission_occ, commission_occ_low, commission_occ_high = bootstrap_roi(
            commission,
            occipital
        )

    front_peaks, front_features = latency_amplitude_peaks(
        "FRONT",
        correct_go_front,
        correct_nogo_front,
        omission_front,
        commission_front,
        times
    )

    occ_peaks, occ_features = latency_amplitude_peaks(
        "OCC",
        correct_go_occ,
        correct_nogo_occ,
        omission_occ,
        commission_occ,
        times
    )

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(14,6),
        sharex=True,
        sharey=True
    )

    plot_bootstrap(
        axs[0],
        times,
        correct_go_front,
        correct_go_front_low,
        correct_go_front_high,
        label=f"Correct Go (n={len(correct_go)})",
        color="tab:blue"
    )

    plot_bootstrap(
        axs[0],
        times,
        correct_nogo_front,
        correct_nogo_front_low,
        correct_nogo_front_high,
        label=f"Correct No-Go (n={len(correct_nogo)})",
        color="tab:orange"
    )

    if plot_omission:

        plot_bootstrap(
            axs[0],
            times,
            omission_front,
            omission_front_low,
            omission_front_high,
            label=f"Omission (n={len(omission)})",
            color="tab:green"
        )

    if plot_commission:

        plot_bootstrap(
            axs[0],
            times,
            commission_front,
            commission_front_low,
            commission_front_high,
            label=f"Commission (n={len(commission)})",
            color="tab:red"
        )

    plot_peak(axs[0], front_peaks["FRONT_P2_Go"], "P2", "tab:blue")
    plot_peak(axs[0], front_peaks["FRONT_N2_Go"], "N2", "tab:blue")
    plot_peak(axs[0], front_peaks["FRONT_P3_Go"], "P3", "tab:blue")

    plot_peak(axs[0], front_peaks["FRONT_P2_NoGo"], "P2", "tab:orange")
    plot_peak(axs[0], front_peaks["FRONT_N2_NoGo"], "N2", "tab:orange")
    plot_peak(axs[0], front_peaks["FRONT_P3_NoGo"], "P3", "tab:orange")

    if plot_omission:

        plot_peak(axs[0], front_peaks["FRONT_P2_Omission"], "P2", "tab:green")
        plot_peak(axs[0], front_peaks["FRONT_N2_Omission"], "N2", "tab:green")
        plot_peak(axs[0], front_peaks["FRONT_P3_Omission"], "P3", "tab:green")

    if plot_commission:

        plot_peak(axs[0], front_peaks["FRONT_P2_Commission"], "P2", "tab:red")
        plot_peak(axs[0], front_peaks["FRONT_N2_Commission"], "N2", "tab:red")
        plot_peak(axs[0], front_peaks["FRONT_P3_Commission"], "P3", "tab:red")

    axs[0].set_title("Frontal")

    plot_bootstrap(
        axs[1],
        times,
        correct_go_occ,
        correct_go_occ_low,
        correct_go_occ_high,
        label=f"Correct Go (n={len(correct_go)})",
        color="tab:blue"
    )

    plot_bootstrap(
        axs[1],
        times,
        correct_nogo_occ,
        correct_nogo_occ_low,
        correct_nogo_occ_high,
        label=f"Correct No-Go (n={len(correct_nogo)})",
        color="tab:orange"
    )

    if plot_omission:

        plot_bootstrap(
            axs[1],
            times,
            omission_occ,
            omission_occ_low,
            omission_occ_high,
            label=f"Omission (n={len(omission)})",
            color="tab:green"
        )

    if plot_commission:

        plot_bootstrap(
            axs[1],
            times,
            commission_occ,
            commission_occ_low,
            commission_occ_high,
            label=f"Commission (n={len(commission)})",
            color="tab:red"
        )

    plot_peak(axs[1], occ_peaks["OCC_P2_Go"], "P2", "tab:blue")
    plot_peak(axs[1], occ_peaks["OCC_N2_Go"], "N2", "tab:blue")
    plot_peak(axs[1], occ_peaks["OCC_P3_Go"], "P3", "tab:blue")

    plot_peak(axs[1], occ_peaks["OCC_P2_NoGo"], "P2", "tab:orange")
    plot_peak(axs[1], occ_peaks["OCC_N2_NoGo"], "N2", "tab:orange")
    plot_peak(axs[1], occ_peaks["OCC_P3_NoGo"], "P3", "tab:orange")

    if plot_omission:

        plot_peak(axs[1], occ_peaks["OCC_P2_Omission"], "P2", "tab:green")
        plot_peak(axs[1], occ_peaks["OCC_N2_Omission"], "N2", "tab:green")
        plot_peak(axs[1], occ_peaks["OCC_P3_Omission"], "P3", "tab:green")

    if plot_commission:

        plot_peak(axs[1], occ_peaks["OCC_P2_Commission"], "P2", "tab:red")
        plot_peak(axs[1], occ_peaks["OCC_N2_Commission"], "N2", "tab:red")
        plot_peak(axs[1], occ_peaks["OCC_P3_Commission"], "P3", "tab:red")

    axs[1].set_title("Occipital")

    for ax in axs:

        ax.axvline(
            0,
            color="k",
            linestyle="--"
        )

        ax.axhline(
            0,
            color="gray",
            linewidth=0.8
        )

        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.3)
        ax.invert_yaxis()

    axs[0].set_ylabel("Amplitude (V)")
    axs[0].legend()

    plt.tight_layout()

    fig.savefig(
        Path(
            results_EEG_path,
            f"ID{id}",
            f"ERP_COMPARE_ID{id}_{date}_{phase}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

def reject_bad_erp_epochs(
    epochs,
    amp_threshold=150e-6,
    step_threshold=50e-6
): # in base of the paper I AM GONNA REJECT BAD EPOCHS 

    data = epochs.get_data()
    bad_epochs = []

    for i in range(data.shape[0]):

        epoch_data = data[i]

        max_amp = np.max(np.abs(epoch_data))

        max_step = np.max(
            np.abs(
                np.diff(
                    epoch_data,
                    axis=1
                )
            )
        )

        if (
            max_amp > amp_threshold
            or
            max_step > step_threshold
        ):

            bad_epochs.append(i)

    print(f"Bad ERP epochs: {len(bad_epochs)} / {len(epochs)}")

    epochs_clean = epochs.drop(
        bad_epochs,
        reason="ERP_THRESHOLD_REJECTION"
    )

    return epochs_clean


if __name__ == '__main__':
    df = load_dataset()
    #preprocess_pipeline()
    #create_dataframe()
    #create_epoch_feature_dataframe()
    #main(df)
    #annottation_df, mne_object = inspect_annotations_clean_data(row = 0)
    #print(annottation_df[annottation_df["description"] == "1003"])
    #plot_time_frequency_analysis(df, row =0)
    #------------------------------------------------------------------------
    df_results_eeg = clean_dataframe_results(pd.read_csv(final_dataframe_results))
    df_results_per_trial = pd.read_csv(final_dataframe_behavior_per_trials) # Important dataset 
    print(df_results_per_trial)
    #------------------------------------------------------------------------
    plot_epoch_go_no_go_per_all_id()
    #for row in range(df.shape[0]):
    #    plot_epoch_per_region(row=row)






#raw = mne.io.read_raw_fif(info_df.loc[0, "MNE PATH"], preload=False)
#print(raw.ch_names)


# ica_path = df.loc[0, "ICA PATH"]
# print("PATH:")
# print(ica_path)
# raw = mne.io.read_raw_fif(ica_path, preload=False)
# raw.plot()
# plt.show(block = True)