from apav import RoiRectPrism
import itertools
import random
from copy import deepcopy
import numpy as np
import apav
import h5py
from pathlib import Path

# Use max Da limit as 307.2 Da, cuz highest seen in data is 298, and 30720 is a multiple of 2.


class Augmentation:
    """
    read the raw source files, apply augmentation, and save as h5 files
    remove_thin: whether remove the original ranges that are thinner that the bin_width, here 0.01da chosen
    """

    def __init__(self, apt_file, ranging_file, savepath, bin_width=0.01, expand_factor=100, shift_range=(1, 10),
                 norm=True, remove_thin=True):
        self.bin_width = bin_width
        self.apt_file = apt_file
        self.ranging_file = ranging_file
        self.norm = norm
        self.expand_factor = expand_factor
        self.shift_range = shift_range
        self.savepath = savepath
        self.remove_thin = remove_thin

    def load_voxel_spectrum(self):
        """
        return spectrum array of different voxel size, shape [307.2/bin_width, N]
        """
        all_y = []
        all_ratio = []
        d = apav.load_apt(self.apt_file)
        ratiolist = np.array(range(4, 10)) * 0.1
        slice_rois = []

        for ratio_of_extent in ratiolist:
            how_many = 2
            X_range = [d.xyz_extents[0][0] * (1 - ratio_of_extent),
                       d.xyz_extents[0][1] * (1 - ratio_of_extent)]
            X_numbers = [it * 0.01 * (X_range[1] - X_range[0]) + X_range[0] for it in
                         random.sample(range(100), how_many)]  # non-repeat numbers within the range
            Y_range = [d.xyz_extents[1][0] * (1 - ratio_of_extent),
                       d.xyz_extents[1][1] * (1 - ratio_of_extent)]
            Y_numbers = [it * 0.01 * (Y_range[1] - Y_range[0]) + X_range[0] for it in
                         random.sample(range(100), how_many)]  # non-repeat numbers within the range
            Z_range = [d.xyz_extents[2][0] * (1 - ratio_of_extent),
                       d.xyz_extents[2][1] * (1 - ratio_of_extent)]
            Z_numbers = [it * 0.01 * (Z_range[1] - Z_range[0]) + X_range[0] for it in
                         random.sample(range(100), how_many)]  # non-repeat numbers within the range

            slice_rois = []
            for (i, j, l) in itertools.product(X_numbers, Y_numbers, Z_numbers):
                width = d.dimensions * ratio_of_extent
                slice_rois.append(RoiRectPrism(d, (i, j, l), width))
                all_ratio.append(ratio_of_extent)

        # for full volume
        slice_rois.append(d)
        all_ratio.append(1)

        all_ratio_ = []
        for i, slice in enumerate(slice_rois):
            x, y = slice.mass_histogram(bin_width=self.bin_width, lower=0, upper=307.2, multiplicity='all', norm=False, )
            # y_log = np.log(y + 1) # leave the log for later
            if y.max() > 1:
                all_y.append(y)
                all_ratio_.append(all_ratio[i])

        return  np.asarray(all_y).T

    def load_ranging(self):
        """
        return ranging label array of a dataset, shape [M, 2], first colum center and second width.
        save normalization after other augmentation
        """
        range_data = apav.RangeCollection.from_rrng(self.ranging_file)
        if self.remove_thin:
            peakwidth = np.array([r.upper - r.lower for r in range_data.ranges])
            rng_array = np.array(range_data.ranges)
            range_data = apav.RangeCollection(rng_array[np.where(peakwidth > self.bin_width)])

        peakwidth = np.array([r.upper - r.lower for r in range_data.ranges])
        peakcenter = np.array([(r.upper + r.lower) * 0.5 for r in range_data.ranges])
        peaks = np.vstack((peakcenter, peakwidth)).T
        ions = np.array([r.ion.formula for r in range_data.ranges])

        return peaks, ions

    def apply_peakshift(self, spectrum, ranges, da_insert=5):
        """
        add random noise points into the spectrum to mimic peak position offset
        Args:
        spectrum: sinle spectrum [n_points, 1]
        ranges: [n_peaks, 2]
        da_insert: defines number of inset points, the maximum peak shift

        returns: new spectrum and new ranges
        """
        # avoid inserting around ranges, +/- 10*bin=0.1Da around them
        remove = []
        total_points = spectrum.shape[0]
        for p in ranges:
            low_index = int(p[0] - p[1] / 2)
            high_index = int(p[0] + p[1] / 2)
            remove = remove + list(range(max(0, low_index - 10), min(high_index + 10, total_points)))
        num_insert = int(da_insert / self.bin_width)  # self.
        max_peak_index = int(ranges.max() / self.bin_width)  # self.

        target_index_range = list(range(0, max_peak_index))
        target_index_range = [index for index in target_index_range if index not in remove]

        # update the spectrum
        target_index = random.sample(target_index_range, num_insert)
        value = []
        for tp in target_index:
            # the points will be inserted behind these target indexes and and use mean value of index and index+1
            value.append(0.5 * (spectrum[tp] + spectrum[tp + 1]))

        i_v_pair = np.vstack((np.array(target_index), np.array(value))).T
        i_v_pair = i_v_pair[i_v_pair[:, 0].argsort()]
        i_v_pair = i_v_pair[::-1, :]  # roll over so the order is descending
        new_spectrum = np.insert(spectrum, np.int_(i_v_pair[:, 0]), i_v_pair[:, 1], axis=0)

        # update the ranges
        new_ranges = deepcopy(ranges)
        for i, (rr_c, _) in enumerate(ranges):
            points_before = i_v_pair[:, 0][i_v_pair[:, 0] < int(rr_c / self.bin_width)].shape[0]
            new_ranges[i][0] = rr_c + points_before * self.bin_width
        return new_spectrum[:total_points], new_ranges

    def normalize(self, spectrum, ranges):
        return (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min()), ranges / 307.2

    def file2h5(self):
        spectrum_voxels = self.load_voxel_spectrum()
        rrs, ions = self.load_ranging()
        spectrum_o_all =[]
        spectrum_log_all = []
        range_o_all = []
        for v in range(spectrum_voxels.shape[-1]):
            da_insert = np.random.random_sample((self.expand_factor,))
            top = min(307-rrs.max(), self.shift_range[1])
            da_insert = da_insert * (top - self.shift_range[0]) + self.shift_range[0]
            for k, it in enumerate(da_insert):
                spectrum_o, range_o = self.apply_peakshift(spectrum_voxels[:, v], rrs, da_insert=it)
                spectrum_log = np.log(spectrum_o+1)
                if self.norm:
                    # apply norm for the log spectrum for yolo-1d
                    spectrum_log, range_o = self.normalize(spectrum_log, range_o)
                spectrum_o_all.append(spectrum_o)
                spectrum_log_all.append(spectrum_log)
                range_o_all.append(range_o)
        spectrum_o_all = np.stack(spectrum_o_all, dtype='float32')
        spectrum_log_all = np.stack(spectrum_log_all, dtype='float32')
        range_o_all = np.stack(range_o_all, dtype='float32')
        print("Augmentation finished for", Path(self.apt_file).stem)
        with h5py.File(self.savepath + Path(self.apt_file).stem + '.h5', 'w') as f:
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('ion', data=np.array(ions,dtype=object), dtype=dt)
            f.create_dataset('if_norm', data=np.array(self.norm))
            f.create_dataset('if_peak_shift', data=np.array(True))
            f.create_dataset('input', data=spectrum_log_all)
            f.create_dataset('label', data=range_o_all)
            f.create_dataset('non_log_spectrums', data=spectrum_o_all)

        print("h5 file writting finished")
