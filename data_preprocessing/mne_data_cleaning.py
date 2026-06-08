import mne
from pathlib import Path
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from collections import Counter
from mne.time_frequency import tfr_array_morlet


mne_files_path = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Raw_data_eeg_psychopy_trimmed")
info_datasets = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Important_datasets\all_ids.csv")
segments_path = Path(r"G:\Mon Drive\M2_Project_Master\Data\Participants data\Segments")


def load_dataset():

    info_df = pd.read_csv(info_datasets)

    if "BAD CHANNELS" in info_df.columns:
        info_df["BAD CHANNELS"] = info_df["BAD CHANNELS"].astype(str)

    if "ICA PATH" in info_df.columns:
        info_df["ICA PATH"] = info_df["ICA PATH"].astype(str)

    if "ICA EXCLUDE" in info_df.columns:
        info_df["ICA EXCLUDE"] = info_df["ICA EXCLUDE"].astype(str)

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

        blink_epochs.average().plot()
        plt.show()
    else:
        print("No blink marker 101 found.")

    ica = mne.preprocessing.ICA(
    n_components=0.99,
    random_state=97,
    max_iter="auto")
    ica.fit(raw, reject_by_annotation=True)
    ica.plot_components()
    plt.show()
    ica.plot_sources(raw)
    plt.show()
    if blink_epochs is not None: 
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
    annotations = get_markers_annotations(id = id, date = date, phase = phase)
    raw = mne.io.read_raw_fif(mne_path, preload=True)
    raw.set_annotations(annotations)

    if "BAD CHANNELS" in df.columns:
        bads = df.loc[row, "BAD CHANNELS"]
        if pd.notna(bads) and bads != "":
            raw.info["bads"] = [ch.strip() for ch in bads.split(",")]

    raw.filter(l_freq = 0.5,h_freq = 40) 
    raw.notch_filter(50)
    if len(raw.info["bads"]) < len(raw.ch_names):
        raw.set_eeg_reference("average")
    else:
        print("All channels marked as bad.")
        return None

    #raw.set_eeg_reference("average")
    #print(raw.info["bads"])
    #picks_channels = ["O1", "O2"]
    return raw

def visualize_signal(df, row, clean = False):
    mne_path = df["MNE PATH"][row]
    date = df["DATE"][row]
    id = df["ID"][row]
    phase = df["LABEL"][row]
    raw = clean_setup_signal(df,row)
    if raw is None: 
        print("Bad session detected. Skipping.")
        return 
    if clean == False: 
        raw.plot(block = False, scalings = "auto",title=f"ID{id} - {date} - {phase}")
        raw.compute_psd().plot()
        raw.plot_sensors(show_names=True)

        if "BAD CHANNELS" not in df.columns:
            df["BAD CHANNELS"] = ""
        raw = get_bad_channels(raw)  
        df.loc[row, "BAD CHANNELS"] = ",".join(raw.info["bads"])
        df.to_csv(info_datasets, index = False)


    else: 
        raw_clean, ica = remove_blink(raw)
        ica_path = save_mne(date, phase, id, data_mne_clean = raw_clean)
        if "ICA PATH" not in df.columns:
            df["ICA PATH"] = ""
        if "ICA EXCLUDE" not in df.columns:
            df["ICA EXCLUDE"] = ""
        else: 
            df["ICA EXCLUDE"] = df["ICA EXCLUDE"].astype(str)

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

def get_fif_path(row):
    expected_name = (f"ID{row['ID']}_{row['DATE']}_{row['LABEL']}_raw.fif")

    return fif_dict.get(expected_name, None)

def get_partipants(): 
    return [2,5,6,8,10,13]

def get_summary(): 
    list_part = get_partipants()
    dataset = pd.read_csv(info_datasets)
    for id in list_part:
        df = dataset[(dataset["ID"] == id) & (dataset["COMPANY"] == f"BitBrain_EEG") ]
        print(f"ID{id}")
        print(Counter(df["LABEL"]))

get_summary()


def preprocessed_subject(info_df, row):
    print(info_df.loc[row, ["ID","DATE","LABEL"]])
    visualize_signal(df=info_df,row=row,clean=True)

def create_epochs(row):
     
    id = row["ID"]
    date = row["DATE"]
    phase = row["LABEL"]

    raw = mne.io.read_raw_fif(row["ICA PATH"],preload=True)

    annotations = get_markers_annotations(
        id=id,
        date=date,
        phase=phase
    )

    raw.set_annotations(annotations)

    ann = raw.annotations

    t90 = ann.onset[ann.description == "90"][0]
    t91 = ann.onset[ann.description == "91"][0]
    t92 = ann.onset[ann.description == "92"][0]
    t93 = ann.onset[ann.description == "93"][0]
    t1000 = ann.onset[ann.description == "1000"]

    margin = 2

    raw_pre = raw.copy().crop(max(0, t90-margin),t91+margin)
    raw_task = raw.copy().crop(max(0, t1000[0]-margin),t1000[-1]+ 90 + margin)
    raw_post = raw.copy().crop(max(0, t92-margin),min(raw.times[-1], t93+margin))

    subject_dir = Path(segments_path, f"ID{id}")
    raw_pre.save(
        Path(subject_dir,
            f"ID{id}_{date}_{phase}_PRE_raw.fif"),
        overwrite=True
    )

    raw_task.save(
        Path(subject_dir,
            f"ID{id}_{date}_{phase}_TASK_raw.fif"),
        overwrite=True
    )

    raw_post.save(
        Path(subject_dir,
            f"ID{id}_{date}_{phase}_POST_raw.fif"),
        overwrite=True
    )
    return raw_pre, raw_task, raw_post

from mne.time_frequency import tfr_array_morlet

def plot_individual_wavelets(raw): 
    freqs = np.arange(1, 40, 1)
    n_cycles = freqs / 2

    data = raw.get_data()

    power = tfr_array_morlet(
        data[np.newaxis, :, :],
        sfreq=raw.info["sfreq"],
        freqs=freqs,
        n_cycles=n_cycles,
        output="power"
    )[0]

    print(power.shape)
    ch = "O2"

    ch_idx = raw.ch_names.index(ch)

    power_db = 10 * np.log10(
        power[ch_idx] + 1e-12
    )
    plt.figure(figsize=(12,6))

    plt.imshow(
        power_db,
        aspect="auto",
        origin="lower",
        extent=[raw.times[0],raw.times[-1],freqs[0],freqs[-1]],
        cmap="viridis"
    )

    plt.colorbar(label="Power (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("O1 - TASK")

    plt.show()
    return False

def visualize_epoch(df, row, raw_pre, raw_task, raw_post):

    epochs = [raw_pre, raw_task, raw_post]
    labels = ["PRE", "TASK", "POST"]
    date = df["DATE"][row]
    id = df["ID"][row]
    phase = df["LABEL"][row]

    for ep, lab in zip(epochs, labels):

        ep.plot(scalings="auto",title=f"ID{id} - {date} - {phase} - {lab}")

    fig, axes = plt.subplots(1, 3, figsize=(15,5), sharey=True)

    for ep, lab, ax in zip(epochs, labels, axes):
        psd = ep.compute_psd(fmax=40)
        psd.plot(axes=ax,show=False)
        ax.set_title(lab)

    fig.suptitle(f"ID{id} - {date} - {phase}",fontsize=16)
    plt.tight_layout()
    plt.show()
    input("Press enter to finish ....")


def compute_wavelet_all_channels(raw,label,row,df):
    date = df["DATE"][row]
    id = df["ID"][row]
    phase = df["LABEL"][row]
    freqs = np.arange(1, 40, 1)
    n_cycles = freqs / 2

    data = raw.get_data()

    power = tfr_array_morlet(
        data[np.newaxis, :, :],
        sfreq=raw.info["sfreq"],
        freqs=freqs,
        n_cycles=n_cycles,
        output="power"
    )[0]

    fig, axes = plt.subplots(
        nrows=len(raw.ch_names),
        ncols=1,
        figsize=(15, 20),
        sharex=True
    )

    for ch_idx, ax in enumerate(axes):

        power_db = 10 * np.log10(
            power[ch_idx] + 1e-12
        )

        im = ax.imshow(
            power_db,
            aspect="auto",
            origin="lower",
            extent=[
                raw.times[0],
                raw.times[-1],
                freqs[0],
                freqs[-1]
            ],
            cmap="viridis"
        )

        ax.set_ylabel(
            raw.ch_names[ch_idx]
        )

        cbar = fig.colorbar(
            im,
            ax=ax
        )

        cbar.set_label("dB")

    axes[-1].set_xlabel("Time (s)")

    #fig.suptitle(f"ID{id} - {date} - {phase} - {label} PHASE",fontsize=12)
    fig.canvas.manager.set_window_title(f"Wavelet | ID{id} | {date} | {phase} | {label} PHASE")

    plt.tight_layout()
    plt.show()
#visualize_epoch(df = info_df, row = row, raw_pre= raw_pre, raw_task = raw_task, raw_post = raw_post)


fif_dict = {}

for f in mne_files_path.rglob("*.fif"):
    fif_dict[f.name] = str(f)

print(f"Found {len(fif_dict)} FIF files")

#MAIN-----------------------------------------------------------------------------------
info_df = load_dataset()
#preprocessed_subject(info_df=info_df, row=3)
row =54 
print(info_df.loc[row, ["ID","DATE","LABEL"]])
raw_pre, raw_task, raw_post = create_epochs(info_df.loc[row,:])
print(f"raw_pre: {raw_pre.annotations} ,\n raw_during: {raw_task.annotations},\n raw_post: {raw_post.annotations}")
print("PRE duration :", raw_pre.times[-1])
print("TASK duration:", raw_task.times[-1])
print("POST duration:", raw_post.times[-1])
#print(type(raw_task.annotations))


raw = raw_post
label = "POST"
compute_wavelet_all_channels(raw,label,row,df = info_df)
input("Press space to continue")

#print(info_df.iloc[54,:].iloc[0])
#print(info_df.iloc[54,:].iloc[1])
#print(info_df.iloc[54,:].iloc[2])



#print(len(info_df))
#print(info_df.index)
#print(info_df.shape)

