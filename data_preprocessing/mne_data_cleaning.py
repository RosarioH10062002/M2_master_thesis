import mne
from pathlib import Path
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter

mne_files_path = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Raw_data_eeg_psychopy_trimmed")
info_datasets = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Important_datasets\all_ids.csv")


def load_dataset():

    info_df = pd.read_csv(info_datasets)

    info_df["PSYCHOPY"] = (info_df["PSYCHOPY"].str.replace("/content/drive/My Drive/", r"G:/Mon Drive/", regex=False))

    info_df["TRIMMED RAW EEG"] = (info_df["TRIMMED RAW EEG"].str.replace( "/content/drive/My Drive/",r"G:/Mon Drive/",regex=False))

    info_df["RAW EEG"] = (info_df["RAW EEG"].str.replace("/content/drive/My Drive/",r"G:/Mon Drive/",regex=False))

    info_df["MNE PATH"] = info_df.apply(get_fif_path,axis=1)

    return info_df

#print(list_path)

def get_bad_channels(raw): 
    bad_channels = input("Bad channels (comma separated) : ")
    raw.info["bads"]= [ch.strip() for ch in bad_channels.split(",") if ch.strip() !=""]
    return raw
def remove_blink(raw): 
    events, event_id = mne.events_from_annotations(raw)

    blink_epochs = mne.Epochs(
        raw,
        events,
        event_id={"blink": event_id["101"]},
        tmin=-0.5,
        tmax=0.8,
        baseline=(-0.5, 0),
        preload=True
    )

    blink_epochs.average().plot()
    plt.show()

    ica = mne.preprocessing.ICA(
    n_components=0.99,
    random_state=97,
    max_iter="auto")
    ica.fit(raw, reject_by_annotation=True)
    ica.plot_components()
    plt.show()
    ica.plot_sources(raw)
    plt.show()
    ica_sources = ica.get_sources(blink_epochs)
    ica_sources.average(picks="all").plot()
    plt.show()
    component = int(input("ICA component to remove: "))
    ica.exclude = [component]
    
    raw_clean = raw.copy()
    ica.apply(raw_clean)
    return raw_clean, ica

def clean_setup_signal(df,row):
    mne_path = df["MNE PATH"][row]
    date = df["DATE"][row]
    id = df["ID"][row]
    phase = df["LABEL"][row]
    annotations = get_markers_annotations(id = id, date = date)
    raw = mne.io.read_raw_fif(mne_path, preload=True)
    raw.set_annotations(annotations)

    if "BAD CHANNELS" in df.columns:
        bads = df.loc[row, "BAD CHANNELS"]
        if pd.notna(bads) and bads != "":
            raw.info["bads"] = [ch.strip() for ch in bads.split(",")]

    raw.filter(l_freq = 0.5,h_freq = 40) 
    raw.notch_filter(50)
    raw.set_eeg_reference("average")
    #print(raw.info["bads"])
    #picks_channels = ["O1", "O2"]
    return raw

def visualize_signal(df, row, clean = False):
    mne_path = df["MNE PATH"][row]
    date = df["DATE"][row]
    id = df["ID"][row]
    phase = df["LABEL"][row]
    raw = clean_setup_signal(df,row)
    if clean == False: 
        if "BAD CHANNELS" not in df.columns:
            df["BAD CHANNELS"] = ""
        raw = get_bad_channels(raw)  
        df.loc[row, "BAD CHANNELS"] = ",".join(raw.info["bads"])
        df.to_csv(info_datasets, index = False)

        raw.plot(block = False, scalings = "auto",title=f"ID{id} - {date} - {phase}")
        raw.compute_psd().plot()
        raw.plot_sensors(show_names=True)
    else: 
        raw_clean, ica = remove_blink(raw)
        ica_path = save_mne(date, phase, id, data_mne_clean = raw_clean)
        if "ICA PATH" not in df.columns:
            df["ICA PATH"] = ""
        if "ICA EXCLUDE" not in df.columns:
            df["ICA EXCLUDE"] = ""

        df.loc[row, "ICA PATH"] = ica_path
        df.loc[row, "ICA EXCLUDE"] = ",".join(map(str, ica.exclude))
        df.to_csv(info_datasets, index = False)

        raw_clean.plot(block = False, scalings = "auto",title=f"ID{id} - {date} - {phase}")
        raw_clean.compute_psd().plot()
        raw_clean.plot_sensors(show_names=True)
    
    input("Enter to continue ...")

def save_mne(date, phase, id, data_mne_clean):
    save_path = Path(mne_files_path,f"ID{id}",f"ID{id}_{date}_{phase}_ica_raw.fif")
    data_mne_clean.save(save_path,overwrite=True)
    print(f"Save: {save_path}")
    return str(save_path)


def get_markers_annotations(id, date):
    for f in Path(mne_files_path).rglob(f"*markers.csv"): 
        if (f"ID{id}_" in f.name and f"_{date}_" in f.name):
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

def get_fif_path(row):
    expected_name = (f"ID{row['ID']}_{row['DATE']}_{row['LABEL']}_raw.fif")

    return fif_dict.get(expected_name, None)

fif_dict = {}

for f in mne_files_path.rglob("*.fif"):
    fif_dict[f.name] = str(f)

print(f"Found {len(fif_dict)} FIF files")

info_df = load_dataset()
print(info_df.iloc[54,:].iloc[0])
print(info_df.iloc[54,:].iloc[1])
print(info_df.iloc[54,:].iloc[2])

visualize_signal(df = info_df, row = 54, clean = True)
