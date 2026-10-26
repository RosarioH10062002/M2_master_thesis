import numpy as np
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from pylsl import StreamInfo, StreamOutlet

# Initial configuration
params = BrainFlowInputParams()
params.serial_port = "/dev/ttyUSB0"

board_id = BoardIds.CYTON_BOARD.value
board = BoardShim(board_id, params)

# Get static board info
sampling_rate = BoardShim.get_sampling_rate(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)

print(f"Sampling rate: {sampling_rate}")
print(f"EEG channels: {eeg_channels}")

# Create EEG LSL stream
info = StreamInfo(
    name='OpenBCI_EEG',
    type='EEG',
    channel_count=len(eeg_channels),
    nominal_srate=sampling_rate,
    channel_format='float32',
    source_id='openbci_cyton_eeg'
)

eeg_outlet = StreamOutlet(info)

all_data = []

try:
    board.prepare_session()
    print("Board session prepared successfully.")

    board.start_stream()
    print("Streaming EEG...")

    while True:
        data = board.get_current_board_data(10)

        if data.shape[1] > 0:
            sample = data[eeg_channels, -1]
            eeg_outlet.push_sample(sample.tolist())
            all_data.append(sample.copy())

        time.sleep(0.001)

except KeyboardInterrupt:
    print("Stopped by user.")

except Exception as e:
    print("Error:", e)

finally:
    try:
        board.stop_stream()
    except:
        pass

    try:
        board.release_session()
    except:
        pass

    print("Board released properly.")

    all_data = np.array(all_data)
    np.save("eeg_data_all.npy", all_data)
    print(f"Saved EEG data with shape: {all_data.shape}")