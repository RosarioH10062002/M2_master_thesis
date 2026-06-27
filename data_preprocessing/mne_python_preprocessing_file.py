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

mne_files_path = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Raw_data_eeg_psychopy_trimmed")
info_datasets = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Important_datasets\Bitbrain_eeg_sessions.csv")
segments_path = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Segments")
results_EEG_path = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Results")
dataframe_results = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Results\eeg_dataframe.csv")


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


if __name__ == '__main__':
    df = load_dataset()
    #preprocess_pipeline()
    create_dataframe()
    #create_epoch_feature_dataframe()
    #main(df)
    #annottation_df, mne_object = inspect_annotations_clean_data(row = 0)
    #print(annottation_df[annottation_df["description"] == "1003"])
    #plot_time_frequency_analysis(df, row =0)
#raw = mne.io.read_raw_fif(info_df.loc[0, "MNE PATH"], preload=False)
#print(raw.ch_names)
df = pd.read_csv(info_datasets)

# ica_path = df.loc[0, "ICA PATH"]
# print("PATH:")
# print(ica_path)
# raw = mne.io.read_raw_fif(ica_path, preload=False)
# raw.plot()
# plt.show(block = True)