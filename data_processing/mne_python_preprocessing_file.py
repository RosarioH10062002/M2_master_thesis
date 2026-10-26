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
        info_df["BAD CHANNELS"] = (
            info_df["BAD CHANNELS"]
            .fillna("")
            .astype(str)
        )

    if "ICA PATH" in info_df.columns:
        info_df["ICA PATH"] = (
            info_df["ICA PATH"]
            .fillna("")
            .astype(str)
        )

    if "ICA EXCLUDE" in info_df.columns:
        info_df["ICA EXCLUDE"] = (
            info_df["ICA EXCLUDE"]
            .fillna("")
            .astype(str)
        )

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

    annotations = get_markers_annotations(
        id=id,
        date=date,
        phase=phase
    )

    raw.set_annotations(annotations)
    
    #Bad channels 
    if "BAD CHANNELS" in df.columns:
        bads = df.loc[row, "BAD CHANNELS"]
        if (pd.notna(bads) and bads != "" and bads.lower() != "nan"):
            raw.info["bads"] = [ch.strip() for ch in bads.split(",")]

    print("Bad channels:", raw.info["bads"])
    
    # Filtering
    raw.filter(
        l_freq=0.5,
        h_freq=40
    )

    raw.notch_filter(
        freqs=50
    )

    # Average reference
    if len(raw.info["bads"]) < len(raw.ch_names):
        raw.set_eeg_reference("average")
    else:
        print("All channels marked as bad.")
        return None
    return raw
def remove_blink(raw):
    events, event_id = mne.events_from_annotations(raw)
    blink_epochs = None
    if "101" in event_id:
        blink_epochs = mne.Epochs(
            raw,
            events,
            event_id={"blink": event_id["101"]},
            tmin=-0.5,
            tmax=0.8,
            baseline=(-0.5, 0),
            preload=True
        )

        print(f"Blink epochs found: {len(blink_epochs)}")

        blink_epochs.average().plot()
        plt.show()

    else:
        print("No blink marker 101 found.")
    print(f"Bad channels: {raw.info['bads']}")

    ica = mne.preprocessing.ICA(
        n_components=0.99,
        random_state=97,
        max_iter="auto"
    )

    ica.fit(
        raw,
        reject_by_annotation=True
    )

    print(f"ICA components: {ica.n_components_}")

    ica.plot_components()
    plt.show()

    ica.plot_sources(raw)
    plt.show()

    if blink_epochs is not None:

        ica_sources = ica.get_sources(
            blink_epochs
        )

        ica_sources.average(
            picks="all"
        ).plot()

        plt.show()

    components = input(
        "Components to remove (e.g. 0,2 or Enter for none): "
    )

    if components.strip() != "":

        ica.exclude = [
            int(c.strip())
            for c in components.split(",")
        ]

        for comp in ica.exclude:
            print(f"Inspecting IC {comp}")
            ica.plot_properties(
                raw,
                picks=[comp]
            )
            plt.show()
    else:
        ica.exclude = []
    print(f"Excluded components: {ica.exclude}")
    raw_clean = raw.copy()
    ica.apply(raw_clean)
    print("ICA applied.")
    return raw_clean, ica
def visualize_signal(df, row, clean=False):

    date = df.loc[row, "DATE"]
    id = df.loc[row, "ID"]
    phase = df.loc[row, "LABEL"]

    raw = clean_setup_signal(df, row)

    if raw is None:

        print("Bad session detected. Skipping.")

        return

    if clean == False:

        raw.plot(
            block=False,
            scalings="auto",
            title=f"ID{id} - {date} - {phase}"
        )

        raw.compute_psd().plot()

        raw.plot_sensors(
            show_names=True
        )

    else:

        raw_clean, ica = remove_blink(raw)

        ica_path = save_mne(
            date,
            phase,
            id,
            data_mne_clean=raw_clean
        )

        if "ICA PATH" not in df.columns:
            df["ICA PATH"] = ""

        if "ICA EXCLUDE" not in df.columns:
            df["ICA EXCLUDE"] = ""

        df.loc[row, "ICA PATH"] = ica_path

        if ica is not None:

            df.loc[row, "ICA EXCLUDE"] = (
                ",".join(
                    map(str, ica.exclude)
                )
            )

        else:

            df.loc[row, "ICA EXCLUDE"] = ""

        df.to_csv(
            info_datasets,
            index=False
        )

        raw_clean.plot(
            block=False,
            scalings="auto",
            title=f"ID{id} - {date} - {phase}"
        )

        raw_clean.compute_psd().plot()

        raw_clean.plot_sensors(
            show_names=True
        )
    input("Enter to continue ...")

def preprocess_all_sessions():
    info_df = load_dataset()
    info_df = add_bad_channels_column(info_df)

    for row in info_df.index:

        print(info_df.loc[row, ["ID","DATE","LABEL"]])

        if (
            "ICA PATH" in info_df.columns
            and pd.notna(info_df.loc[row, "ICA PATH"])
            and info_df.loc[row, "ICA PATH"] != ""
        ):
            print("Already processed. Skipping.")
            continue

        visualize_signal(
            info_df,
            row,
            clean=True
        )
        
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

#PHASE I 
info_df = load_dataset()
info_df = add_bad_channels_column(info_df)
info_df.to_csv(info_datasets,index=False)
#PHASE II
preprocess_all_sessions()
#info_df

#raw = mne.io.read_raw_fif(info_df.loc[0, "MNE PATH"], preload=False)
#print(raw.ch_names)