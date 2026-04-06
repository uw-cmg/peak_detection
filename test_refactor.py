import os
import detect_peaks_refactor as peaks
import peak_detection.data_io as data_io

apt_file = 'APT_test\\R13_40310Zr Unsaved - Top Level ROI.apt'
rrng_file = 'RRNG_test\\R13_40310Zr Top Level ROI.RRNG'

output_dir = os.path.join(os.getcwd(), 'test')

training_path = 'peak_detection/Ionclassifier/training_data/NewData/Data0001'

result = peaks.process_dataset(
    apt_file, rrng_file, output_dir='test',
    flag_unknowns=True, save_rrng_output=True,
    use_mc_distance=True, mc_threshold=0.2,
    training_path=training_path
)
