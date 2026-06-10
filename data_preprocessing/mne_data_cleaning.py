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
    plt.show(block = False)
#visualize_epoch(df = info_df, row = row, raw_pre= raw_pre, raw_task = raw_task, raw_post = raw_post)

def show_pre_post_wavelet_all_channels(raw_pre,raw_post,row,info_df):
    compute_wavelet_all_channels(raw = raw_pre,label = "PRE",row = row,df = info_df)
    compute_wavelet_all_channels(raw = raw_post,label = "POST",row = row,df = info_df)
    input("Press space to continue")

fif_dict = {}

for f in mne_files_path.rglob("*.fif"):
    fif_dict[f.name] = str(f)

print(f"Found {len(fif_dict)} FIF files")


def compare_wavelets(raw_pre, raw_post,channels,label,row,df):
    date = df["DATE"][row]
    id = df["ID"][row]
    phase = df["LABEL"][row]

    freqs = np.arange(1, 40, 1)
    n_cycles = freqs / 2

    # PRE
    power_pre = tfr_array_morlet(
        raw_pre.get_data()[np.newaxis, :, :],
        sfreq=raw_pre.info["sfreq"],
        freqs=freqs,
        n_cycles=n_cycles,
        output="power"
    )[0]

    # POST
    power_post = tfr_array_morlet(
        raw_post.get_data()[np.newaxis, :, :],
        sfreq=raw_post.info["sfreq"],
        freqs=freqs,
        n_cycles=n_cycles,
        output="power"
    )[0]

    power_pre_db = 10 * np.log10(power_pre + 1e-12)
    power_post_db = 10 * np.log10(power_post + 1e-12)

    vmin = min(
        power_pre_db.min(),
        power_post_db.min()
    )

    vmax = max(
        power_pre_db.max(),
        power_post_db.max()
    )

    fig, axes = plt.subplots(
        nrows = len(channels) * 2,
        ncols=1,
        figsize=(14, 18)
    )

    fig.canvas.manager.set_window_title(
        f"{label} Channels PRE vs POST"
    )

    row_plot = 0

    for ch in channels:

        ch_pre = raw_pre.ch_names.index(ch)
        ch_post = raw_post.ch_names.index(ch)

        # PRE
        im = axes[row_plot].imshow(
            power_pre_db[ch_pre],
            aspect="auto",
            origin="lower",
            extent=[
                raw_pre.times[0],
                raw_pre.times[-1],
                freqs[0],
                freqs[-1]
            ],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax
        )

        axes[row_plot].set_ylabel(
            f"{ch}\nPRE"
        )

        row_plot += 1

        # POST
        axes[row_plot].imshow(
            power_post_db[ch_post],
            aspect="auto",
            origin="lower",
            extent=[
                raw_post.times[0],
                raw_post.times[-1],
                freqs[0],
                freqs[-1]
            ],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax
        )

        axes[row_plot].set_ylabel(
            f"{ch}\nPOST"
        )

        row_plot += 1

    axes[-1].set_xlabel("Time (s)")

    cbar = fig.colorbar(
        im,
        ax=axes,
        location="right",
        fraction=0.02,
        pad=0.02
    )

    cbar.set_label("Power (dB)")
    fig.canvas.manager.set_window_title(f"Wavelet | ID{id} | {date} | {phase} | {label} PHASE")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show(block = False)

def get_frontal_channels(): 
    return ["AF7", "AF8", "Fp1", "Fp2"]

def get_occipital_channels(): 
    return ["P7", "P8", "O1", "O2"]

def show_wavelet_two_zones(raw_pre,raw_post,row,df): 
    channels = get_frontal_channels()
    compare_wavelets(raw_pre=raw_pre,raw_post=raw_post, channels = channels, label = "Frontal",row = row, df = df)
    channels = get_occipital_channels()
    compare_wavelets(raw_pre=raw_pre,raw_post=raw_post, channels = channels, label = "Occipital",row = row, df = df)
    input("Press enter to continue...")

def look_specific_wavelet(): 
    return None 

def region_bandpower(
    raw,
    channels,
    l_freq,
    h_freq,
    smooth_seconds=1
):

    raw_band = raw.copy()

    raw_band.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        verbose=False
    )

    raw_band.apply_hilbert(
        envelope=True
    )

    power = raw_band.get_data()**2

    idx = [
        raw_band.ch_names.index(ch)
        for ch in channels
    ]

    regional_power = power[idx].mean(axis=0)

    sigma = smooth_seconds * raw_band.info["sfreq"]

    regional_power = gaussian_filter1d(
        regional_power,
        sigma=sigma
    )

    return raw_band.times, regional_power

def plot_regions(
    raw,
    l_freq,
    h_freq,
    band_name=""
):

    times, frontal = region_bandpower(
        raw,
        get_frontal_channels(),
        l_freq,
        h_freq
    )

    _, occipital = region_bandpower(
        raw,
        get_occipital_channels(),
        l_freq,
        h_freq
    )

    plt.figure(figsize=(12,5))

    plt.plot(
        times,
        frontal,
        label="Frontal"
    )

    plt.plot(
        times,
        occipital,
        label="Occipital"
    )
    plt.axvspan(0, 62, alpha=0.1, label="Eyes open", color = "yellow")
    plt.axvspan(62, 122, alpha=0.1, label="Eyes closed", color = "blue")


    plt.xlabel("Time (s)")
    plt.ylabel("Power (uV²)")

    plt.title(
        f"{band_name} Power"
    )

    plt.legend()
    plt.tight_layout()
    plt.show(block = False)
def compare_pre_post_regions(
    raw_pre,
    raw_post,
    id,
    date,
    phase,
    l_freq,
    h_freq,
    band_name="Alpha"
):

    times_pre, pre_frontal = region_bandpower(
        raw_pre,
        get_frontal_channels(),
        l_freq,
        h_freq
    )

    _, pre_occipital = region_bandpower(
        raw_pre,
        get_occipital_channels(),
        l_freq,
        h_freq
    )

    times_post, post_frontal = region_bandpower(
        raw_post,
        get_frontal_channels(),
        l_freq,
        h_freq
    )

    _, post_occipital = region_bandpower(
        raw_post,
        get_occipital_channels(),
        l_freq,
        h_freq
    )

    fig, ax = plt.subplots(
        figsize=(14,6)
    )

    ax.plot(
        times_pre,
        pre_frontal,
        label="PRE Frontal"
    )

    ax.plot(
        times_pre,
        pre_occipital,
        label="PRE Occipital"
    )

    ax.plot(
        times_post,
        post_frontal,
        label="POST Frontal"
    )

    ax.plot(
        times_post,
        post_occipital,
        label="POST Occipital"
    )

    ax.axvspan(
        0,
        60,
        alpha=0.1,
        color="yellow"
    )

    ax.axvspan(
        60,
        120,
        alpha=0.1,
        color="lightblue"
    )

    ax.text(
        30,
        ax.get_ylim()[1]*0.95,
        "Eyes Open",
        ha="center"
    )

    ax.text(
        90,
        ax.get_ylim()[1]*0.95,
        "Eyes Closed",
        ha="center"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (V²)")
    ax.set_title(
        f"{band_name} Power\nID{id} | {date} | {phase}"
    )

    ax.legend()

    fig.canvas.manager.set_window_title(
        f"{band_name} PRE vs POST | ID{id} | {date} | {phase}"
    )

    plt.tight_layout()
    plt.show(block=False)
def compare_pre_post_regions_eyes_open(
    raw_pre,
    raw_post,
    id,
    date,
    phase,
    l_freq,
    h_freq,
    band_name="Alpha"
):

    times_pre, pre_frontal = region_bandpower(
        raw_pre,
        get_frontal_channels(),
        l_freq,
        h_freq
    )

    _, pre_occipital = region_bandpower(
        raw_pre,
        get_occipital_channels(),
        l_freq,
        h_freq
    )

    times_post, post_frontal = region_bandpower(
        raw_post,
        get_frontal_channels(),
        l_freq,
        h_freq
    )

    _, post_occipital = region_bandpower(
        raw_post,
        get_occipital_channels(),
        l_freq,
        h_freq
    )

    # Solo Eyes Open
    #mask_pre = times_pre <= 60
    #mask_post = times_post <= 60
    mask_pre = (times_pre >= 2) & (times_pre <= 60)
    mask_post = (times_post >= 2) & (times_post <= 60)

    fig, ax = plt.subplots(
        figsize=(14,6)
    )

    ax.plot(
        times_pre[mask_pre],
        pre_frontal[mask_pre],
        label="PRE Frontal"
    )

    ax.plot(
        times_pre[mask_pre],
        pre_occipital[mask_pre],
        label="PRE Occipital"
    )

    ax.plot(
        times_post[mask_post],
        post_frontal[mask_post],
        label="POST Frontal"
    )

    ax.plot(
        times_post[mask_post],
        post_occipital[mask_post],
        label="POST Occipital"
    )

    ax.axvspan(
        0,
        60,
        alpha=0.1,
        color="yellow",
        label="Eyes Open"
    )

    ymax = ax.get_ylim()[1]

    ax.text(
        30,
        ymax * 0.95,
        "Eyes Open",
        ha="center"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (V²)")

    #ax.set_title(f"{band_name} Power (Eyes Open)\nID{id} | {date} | {phase}")

    ax.legend()

    fig.canvas.manager.set_window_title(
        f"{band_name} PRE vs POST Eyes Open | ID{id} | {date} | {phase}"
    )

    plt.tight_layout()
    plt.show(block=False)

def plot_all_bands_pre_post(
    raw_pre,
    raw_post,
    id,
    date,
    phase
):

    bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Beta": (13, 30),
        "Gamma": (30, 45)
    }

    fig, axes = plt.subplots(
        len(bands),
        1,
        figsize=(15,18),
        sharex=True
    )

    for ax, (band_name, (l_freq, h_freq)) in zip(
        axes,
        bands.items()
    ):

        times_pre, pre_frontal = region_bandpower(
            raw_pre,
            get_frontal_channels(),
            l_freq,
            h_freq
        )

        _, pre_occipital = region_bandpower(
            raw_pre,
            get_occipital_channels(),
            l_freq,
            h_freq
        )

        times_post, post_frontal = region_bandpower(
            raw_post,
            get_frontal_channels(),
            l_freq,
            h_freq
        )

        _, post_occipital = region_bandpower(
            raw_post,
            get_occipital_channels(),
            l_freq,
            h_freq
        )

        mask_pre = (times_pre >= 4) & (times_pre <= 60)
        mask_post = (times_post >= 4) & (times_post <= 60)

        ax.plot(
            times_pre[mask_pre],
            pre_frontal[mask_pre],
            label="PRE Frontal"
        )

        ax.plot(
            times_pre[mask_pre],
            pre_occipital[mask_pre],
            label="PRE Occipital"
        )

        ax.plot(
            times_post[mask_post],
            post_frontal[mask_post],
            label="POST Frontal"
        )

        ax.plot(
            times_post[mask_post],
            post_occipital[mask_post],
            label="POST Occipital"
        )

        ax.set_title(
            f"{band_name} ({l_freq}-{h_freq} Hz)", fontsize = 8
        )
        

        ax.set_ylabel("Power")

        ax.axvspan(
            0,
            60,
            alpha=0.05,
            color="yellow"
        )

    axes[0].legend()

    axes[-1].set_xlabel(
        "Time (s)"
    )

    #fig.suptitle(f"PRE vs POST (Eyes Open)\nID{id} | {date} | {phase}",fontsize=16)

    fig.canvas.manager.set_window_title(
        f"All Bands | ID{id} | {date} | {phase}"
    )

    plt.tight_layout()

    plt.show(block=False)

def compute_bandpower_summary(
    raw_pre,
    raw_post,
    id,
    date,
    phase
):

    bands = {
        "Delta": (0.5,4),
        "Theta": (4,8),
        "Alpha": (8,12),
        "Beta": (13,30),
        "Gamma": (30,45)
    }

    results = {
        "ID": id,
        "DATE": date,
        "PHASE": phase
    }

    for band_name, (l_freq,h_freq) in bands.items():

        # PRE
        times, frontal = region_bandpower(
            raw_pre,
            get_frontal_channels(),
            l_freq,
            h_freq
        )

        _, occipital = region_bandpower(
            raw_pre,
            get_occipital_channels(),
            l_freq,
            h_freq
        )

        mask = (times >= 4) & (times <= 60)

        results[
            f"PRE_Frontal_{band_name}"
        ] = np.median(frontal[mask])

        results[
            f"PRE_Occipital_{band_name}"
        ] = np.median(occipital[mask])

        # POST
        times, frontal = region_bandpower(
            raw_post,
            get_frontal_channels(),
            l_freq,
            h_freq
        )

        _, occipital = region_bandpower(
            raw_post,
            get_occipital_channels(),
            l_freq,
            h_freq
        )

        mask = (times >= 4) & (times <= 60)

        results[
            f"POST_Frontal_{band_name}"
        ] = np.median(frontal[mask])

        results[
            f"POST_Occipital_{band_name}"
        ] = np.median(occipital[mask])

    return results
def build_dataframe(info_df):
    all_results = []

    for row in range(len(info_df)):

        raw_pre, raw_task, raw_post = create_epochs(
            info_df.loc[row,:]
        )

        summary = compute_bandpower_summary(
            raw_pre,
            raw_post,
            id=info_df.loc[row,"ID"],
            date=info_df.loc[row,"DATE"],
            phase=info_df.loc[row,"LABEL"]
        )

        all_results.append(summary)

    bandpower_df = pd.DataFrame(
        all_results
    )
    return bandpower_df

def statistics_pre_post(bandpower_df): 
    A = bandpower_df.loc[
        bandpower_df["PHASE"]=="A",
        "PRE_Occipital_Alpha"
    ]

    B = bandpower_df.loc[
        bandpower_df["PHASE"]=="B",
        "PRE_Occipital_Alpha"
    ]

    return mannwhitneyu(A,B)

def build_dataframe_id(info_df, id):

    all_results = []
    info_df = info_df[info_df["COMPANY"] != "OpenBCI_EEG_eeg"]
    subject_rows = info_df[
        info_df["ID"] == id
    ].index

    for row in subject_rows:

        print(info_df.loc[row, ["ID","DATE","LABEL"]])

        raw_pre, raw_task, raw_post = create_epochs(
            info_df.loc[row,:]
        )

        summary = compute_bandpower_summary(
            raw_pre,
            raw_post,
            id=info_df.loc[row,"ID"],
            date=info_df.loc[row,"DATE"],
            phase=info_df.loc[row,"LABEL"]
        )

        all_results.append(summary)

    return pd.DataFrame(all_results)
def statistics_phase_A_vs_B(
    bandpower_df,
    variable="PRE_Occipital_Alpha"
):

    A = bandpower_df.loc[
        bandpower_df["PHASE"]=="A",
        variable
    ]

    B = bandpower_df.loc[
        bandpower_df["PHASE"]=="B",
        variable
    ]

    print(f"Length A: {len(A)}")
    print(f"Length B: {len(B)}")

    if len(A) == 0 or len(B) == 0:
        print("Missing A or B sessions.")
        return None

    stat, p = mannwhitneyu(A, B)

    print(f"Statistic={stat}")
    print(f"p-value={p}")

    return stat, p

def pipeline(info_df,rowN): 
    print(info_df.loc[row, ["ID","DATE","LABEL"]])
    keyword = input("Visualize? Y/N")
    if keyword == "Y": 
        visualize_signal(info_df, row, clean = False)
    keyword = input("Preprocessed? Y/N")
    if keyword == "Y": 
        preprocessed_subject(info_df=info_df, row=row)

    keyword = input("Create epochs? Y/N")
    if keyword == "Y": 
         raw_pre, raw_task, raw_post = create_epochs(info_df.loc[row,:])
         print(f"raw_pre: {raw_pre.annotations} ,\n raw_during: {raw_task.annotations},\n raw_post: {raw_post.annotations}")
         print("PRE duration :", raw_pre.times[-1])
         print("TASK duration:", raw_task.times[-1])
         print("POST duration:", raw_post.times[-1])
         visualize_epoch(df = info_df, row = row, raw_pre= raw_pre, raw_task = raw_task, raw_post = raw_post)
    keyword = input("Create wavelets? Y/N")
    if keyword == "Y": 
        show_pre_post_wavelet_all_channels(raw_pre,raw_post,row,info_df)
        show_wavelet_two_zones(raw_pre=raw_pre,raw_post = raw_post, row = row, df = info_df)

    keyword = input("Plot band power? Two zonesY/N")
    if keyword == "Y": 
        plot_all_bands_pre_post(
        raw_pre,
        raw_post,
        id=info_df.loc[row,"ID"],
        date=info_df.loc[row,"DATE"],
        phase=info_df.loc[row,"LABEL"])

def pipeline_statistics(id):

    info_df = load_dataset()

    bandpower_df = build_dataframe_id(
        info_df,
        id=id
    )

    variables = [
        "POST_Frontal_Delta",
        "POST_Frontal_Theta",
        "POST_Frontal_Alpha",
        "POST_Frontal_Beta",
        "POST_Frontal_Gamma"
    ]

    all_results = []

    for variable in variables:

        result = statistics_phase_A_vs_B(
            bandpower_df,
            variable=variable
        )

        all_results.append(result)

    results_df = pd.DataFrame(all_results)

    print(results_df)

    return results_df
    '''sns.boxplot(
        data=bandpower_df,
        x="PHASE",
        y="PRE_Occipital_Alpha"
    )

    sns.stripplot(
        data=bandpower_df,
        x="PHASE",
        y="PRE_Occipital_Alpha",
        color="black"
    )'''




#MAIN-----------------------------------------------------------------------------------
info_df = load_dataset()
row = 54
pipeline(info_df,row)
#print(pipeline_statistics(id = 13))
#info_df = load_dataset()
'''row = 20
for row in range(29, 35):

    print(info_df.loc[row, ["ID","DATE","LABEL"]])

    raw_pre, raw_task, raw_post = create_epochs(
        info_df.loc[row,:]
    )

    plot_all_bands_pre_post(
        raw_pre,
        raw_post,
        id=info_df.loc[row,"ID"],
        date=info_df.loc[row,"DATE"],
        phase=info_df.loc[row,"LABEL"]
    )'''
'''plot_regions(
    raw_pre,
    8,
    12,
    "Alpha"
)
plot_regions(
    raw_post,
    8,
    12,
    "Alpha"
)'''
'''compare_pre_post_regions(
    raw_pre,
    raw_post,
    id=info_df.loc[row,"ID"],
    date=info_df.loc[row,"DATE"],
    phase=info_df.loc[row,"LABEL"],
    l_freq=8,
    h_freq=12,
    band_name="Alpha"
)'''
'''compare_pre_post_regions_eyes_open(
    raw_pre,
    raw_post,
    id=info_df.loc[row,"ID"],
    date=info_df.loc[row,"DATE"],
    phase=info_df.loc[row,"LABEL"],
    l_freq=8,
    h_freq=12,
    band_name="Alpha"
)'''
'''plot_all_bands_pre_post(
    raw_pre,
    raw_post,
    id=info_df.loc[row,"ID"],
    date=info_df.loc[row,"DATE"],
    phase=info_df.loc[row,"LABEL"]
)'''
input("Pres space to continue...")
'''#preprocessed_subject(info_df=info_df, row=3)
for row, session in info_df.iterrows():
    #row =54 
    if info_df.loc[row, "COMPANY"] == "OpenBCI_EEG_eeg":
        continue
    else: 
        print(info_df.loc[row, ["ID","DATE","LABEL"]])
        raw_pre, raw_task, raw_post = create_epochs(info_df.loc[row,:])
        print(f"raw_pre: {raw_pre.annotations} ,\n raw_during: {raw_task.annotations},\n raw_post: {raw_post.annotations}")
        print("PRE duration :", raw_pre.times[-1])
        print("TASK duration:", raw_task.times[-1])
        print("POST duration:", raw_post.times[-1])
        #print(type(raw_task.annotations))
        #show_pre_post_wavelet_all_channels(raw_pre,raw_post,row,info_df)
        #print(info_df.iloc[54,:].iloc[0])
        #print(info_df.iloc[54,:].iloc[1])
        #print(info_df.iloc[54,:].iloc[2])
        show_wavelet_two_zones(raw_pre=raw_pre,raw_post = raw_post, row = row, df = info_df)
        #print(len(info_df))
        #print(info_df.index)
        #print(info_df.shape)
'''
