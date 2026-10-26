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


BAD_CHANNELS = {
    (5, "13-05-26", "A"): "AF7",
    (6, "11-05-26", "B"): "O2",
    (6, "13-05-26", "B"): "O1,P7",
    (6, "19-05-26", "B"): "O1,P7,P8,O2",
    (8, "12-05-26", "A"): "AF8",
    (8, "13-05-26", "A"): "P7",
    (8, "18-05-26", "A"): "O1,P7,P8,O2"
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
    
    raw.plot(exclude = "bads")
    
    return raw

def get_psd(mne_object): 
    spectrum = mne_object.compute_psd()
    spectrum.plot(average=True, exclude="bads", amplitude=False)
    plt.show()

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
    eog_idx, scores = ica.find_bads_eog(raw)
    ica.plot_scores(scores, exclude=eog_idx)
    print(f"EOG INDEX: {eog_idx}, SCORES: {scores}")
    ica.plot_components()
    ica.plot_sources(raw)
    plt.show(block = False)
    if len(eog_idx) > 0:
        ica.plot_properties(raw, picks=eog_idx)
    plt.show()

    labels = label_components(raw, ica, method="iclabel")
    for i, (label, probs) in enumerate(zip(labels["labels"], labels["y_pred_proba"])):
        confidence = np.max(probs)
        print(f"IC{i}: {label} ({confidence:.2f})")

    ica_to_exclude = input("ICA COMPONENTS TO BE REMOVED(e.g. [1,2]): ").split(",")
    ica_to_exclude = [int(x.strip()) for x in ica_to_exclude if x.strip() != ""]
    ica.exclude = ica_to_exclude
    print(f"Components excluded: {ica.exclude}")
    return ica


def first_phase(df,row):
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
    ica_path = save_mne(date,phase,id,data_mne_clean=raw_clean)

    if "ICA PATH" not in df.columns:
        df["ICA PATH"] = ""
    if "ICA EXCLUDE" not in df.columns:
        df["ICA EXCLUDE"] = ""
    df.loc[row, "ICA PATH"] = ica_path
    if ica is not None:
        df.loc[row, "ICA EXCLUDE"] = (",".join(map(str, ica.exclude)))
    else:
        df.loc[row, "ICA EXCLUDE"] = ""
    df.to_csv(info_datasets,index=False)

    #Look for the good artifacts
    raw_clean.plot(block=False,scalings="auto",title=f"ID{id} - {date} - {phase}")
    get_psd(mne_object=raw_clean)
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


def main(): 
    df = load_dataset()
    df = add_bad_channels_column(df)
    row = 0
    raw_clean = first_phase(df,row)
    if raw_clean is not None:
        print(raw_clean.ch_names)


if __name__ == '__main__':
    main()
#raw = mne.io.read_raw_fif(info_df.loc[0, "MNE PATH"], preload=False)
#print(raw.ch_names)