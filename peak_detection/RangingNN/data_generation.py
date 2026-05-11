from apav import RoiRectPrism
import itertools
import random
from copy import deepcopy
import numpy as np
import apav
import h5py
from pathlib import Path


# Use max Da limit as 307.2 Da, because highest seen in data is ~298,
# and 307.2 / 0.01 = 30720 bins.
MAX_DA = 307.2


class Augmentation:
    """
    Read raw atom-probe source files, apply augmentation, and save as .h5 files.

    Parameters
    ----------
    apt_file : str or Path
        Path to .pos, .epos, .apt, or .ato file.
    ranging_file : str or Path
        Path to .rrng file.
    savepath : str or Path
        Output directory for .h5 file.
    bin_width : float
        Mass histogram bin width in Da.
    expand_factor : int
        Number of peak-shift augmentations per voxel spectrum.
    shift_range : tuple
        Minimum and maximum peak shift in Da.
    norm : bool
        Whether to normalize log spectra and ranges.
    remove_thin : bool
        Whether to remove original ranges thinner than bin_width.
    """

    def __init__(
        self,
        apt_file,
        ranging_file,
        savepath,
        bin_width=0.01,
        expand_factor=100,
        shift_range=(1, 10),
        norm=True,
        remove_thin=True,
    ):
        self.bin_width = bin_width
        self.apt_file = Path(apt_file)
        self.ranging_file = Path(ranging_file)
        self.norm = norm
        self.expand_factor = expand_factor
        self.shift_range = shift_range
        self.savepath = Path(savepath)
        self.remove_thin = remove_thin

    def load_roi(self):
        """
        Load atom-probe data using the correct APAV reader based on file extension.

        This avoids sending .POS files to apav.load_apt(), which can trigger
        UnicodeDecodeError when the APT reader tries to parse POS binary data.
        """
        path = str(self.apt_file)
        suffix = self.apt_file.suffix.lower().strip()

        print(f"Loading atom-probe file: {path}")
        print(f"Detected extension: {suffix}")

        if suffix == ".pos":
            # Prefer Roi.from_pos if available; fall back to apav.load_pos.
            if hasattr(apav, "Roi") and hasattr(apav.Roi, "from_pos"):
                return apav.Roi.from_pos(path)
            elif hasattr(apav, "load_pos"):
                return apav.load_pos(path)
            else:
                raise AttributeError("Could not find APAV POS loader: Roi.from_pos or load_pos")

        elif suffix == ".epos":
            if hasattr(apav, "Roi") and hasattr(apav.Roi, "from_epos"):
                return apav.Roi.from_epos(path)
            elif hasattr(apav, "load_epos"):
                return apav.load_epos(path)
            else:
                raise AttributeError("Could not find APAV ePOS loader: Roi.from_epos or load_epos")

        elif suffix == ".apt":
            if hasattr(apav, "load_apt"):
                return apav.load_apt(path)
            elif hasattr(apav, "Roi") and hasattr(apav.Roi, "from_apt"):
                return apav.Roi.from_apt(path)
            else:
                raise AttributeError("Could not find APAV APT loader: load_apt or Roi.from_apt")

        elif suffix == ".ato":
            if hasattr(apav, "Roi") and hasattr(apav.Roi, "from_ato"):
                return apav.Roi.from_ato(path)
            elif hasattr(apav, "load_ato"):
                return apav.load_ato(path)
            else:
                raise AttributeError("Could not find APAV ATO loader: Roi.from_ato or load_ato")

        else:
            raise ValueError(f"Unsupported atom-probe file extension: {suffix}")

    def load_voxel_spectrum(self):
        """
        Return spectrum array for different voxel sizes.

        Output shape:
            [MAX_DA / bin_width, N_voxel_spectra]
        """
        all_y = []
        all_ratio = []

        d = self.load_roi()

        ratiolist = np.array(range(4, 10)) * 0.1
        slice_rois = []  # Do not reset inside the loop.

        for ratio_of_extent in ratiolist:
            how_many = 2

            x_range = [
                d.xyz_extents[0][0] * (1 - ratio_of_extent),
                d.xyz_extents[0][1] * (1 - ratio_of_extent),
            ]
            x_numbers = [
                it * 0.01 * (x_range[1] - x_range[0]) + x_range[0]
                for it in random.sample(range(100), how_many)
            ]

            y_range = [
                d.xyz_extents[1][0] * (1 - ratio_of_extent),
                d.xyz_extents[1][1] * (1 - ratio_of_extent),
            ]
            y_numbers = [
                it * 0.01 * (y_range[1] - y_range[0]) + y_range[0]
                for it in random.sample(range(100), how_many)
            ]

            z_range = [
                d.xyz_extents[2][0] * (1 - ratio_of_extent),
                d.xyz_extents[2][1] * (1 - ratio_of_extent),
            ]
            z_numbers = [
                it * 0.01 * (z_range[1] - z_range[0]) + z_range[0]
                for it in random.sample(range(100), how_many)
            ]

            for center in itertools.product(x_numbers, y_numbers, z_numbers):
                width = d.dimensions * ratio_of_extent
                slice_rois.append(RoiRectPrism(d, center, width))
                all_ratio.append(ratio_of_extent)

        # Add full volume.
        slice_rois.append(d)
        all_ratio.append(1.0)

        all_ratio_kept = []

        for i, roi_slice in enumerate(slice_rois):
            x, y = roi_slice.mass_histogram(
                bin_width=self.bin_width,
                lower=0,
                upper=MAX_DA,
                multiplicity="all",
                norm=False,
            )

            if y.max() > 1:
                all_y.append(y)
                all_ratio_kept.append(all_ratio[i])

        if len(all_y) == 0:
            raise ValueError(f"No valid voxel spectra generated for {self.apt_file}")

        return np.asarray(all_y).T

    def load_ranging(self):
        """
        Return ranging labels.

        Returns
        -------
        peaks : np.ndarray
            Shape [M, 2], where first column is peak center and second is width.
        ions : np.ndarray
            Ion formula labels.
        """
        range_data = apav.RangeCollection.from_rrng(str(self.ranging_file))

        if self.remove_thin:
            peakwidth = np.array([r.upper - r.lower for r in range_data.ranges])
            rng_array = np.array(range_data.ranges, dtype=object)
            keep = np.where(peakwidth > self.bin_width)[0]
            range_data = apav.RangeCollection(rng_array[keep])

        peakwidth = np.array([r.upper - r.lower for r in range_data.ranges])
        peakcenter = np.array([(r.upper + r.lower) * 0.5 for r in range_data.ranges])

        peaks = np.vstack((peakcenter, peakwidth)).T
        ions = np.array([r.ion.formula for r in range_data.ranges], dtype=str)

        return peaks, ions

    def apply_peakshift(self, spectrum, ranges, da_insert=5):
        """
        Add random inserted points into the spectrum to mimic peak-position offset.

        Parameters
        ----------
        spectrum : np.ndarray
            Single spectrum, shape [n_points].
        ranges : np.ndarray
            Peak ranges, shape [n_peaks, 2], columns are [center_Da, width_Da].
        da_insert : float
            Approximate maximum inserted Da shift.

        Returns
        -------
        new_spectrum : np.ndarray
            Shifted spectrum with original length.
        new_ranges : np.ndarray
            Shifted ranges.
        """
        spectrum = np.asarray(spectrum)
        ranges = np.asarray(ranges)

        remove = []
        total_points = spectrum.shape[0]

        # Exclude insertion near known ranges.
        # Convert Da to bin index.
        for center_da, width_da in ranges:
            low_index = int((center_da - width_da / 2) / self.bin_width)
            high_index = int((center_da + width_da / 2) / self.bin_width)

            remove.extend(
                range(
                    max(0, low_index - 10),
                    min(high_index + 10, total_points),
                )
            )

        num_insert = int(da_insert / self.bin_width)

        # Only insert before the largest relevant peak.
        max_peak_index = int(ranges[:, 0].max() / self.bin_width)
        max_peak_index = min(max_peak_index, total_points - 2)

        target_index_range = list(range(0, max_peak_index))
        remove_set = set(remove)
        target_index_range = [idx for idx in target_index_range if idx not in remove_set]

        if len(target_index_range) < num_insert:
            raise ValueError(
                f"Not enough valid insertion points. Requested {num_insert}, "
                f"available {len(target_index_range)} for {self.apt_file}"
            )

        target_index = random.sample(target_index_range, num_insert)

        value = []
        for tp in target_index:
            value.append(0.5 * (spectrum[tp] + spectrum[tp + 1]))

        i_v_pair = np.vstack((np.array(target_index), np.array(value))).T
        i_v_pair = i_v_pair[i_v_pair[:, 0].argsort()]
        i_v_pair = i_v_pair[::-1, :]  # descending order

        new_spectrum = np.insert(
            spectrum,
            np.int_(i_v_pair[:, 0]),
            i_v_pair[:, 1],
            axis=0,
        )

        new_ranges = deepcopy(ranges)

        for i, (range_center_da, _) in enumerate(ranges):
            range_center_index = int(range_center_da / self.bin_width)
            points_before = i_v_pair[:, 0][i_v_pair[:, 0] < range_center_index].shape[0]
            new_ranges[i][0] = range_center_da + points_before * self.bin_width

        return new_spectrum[:total_points], new_ranges

    def normalize(self, spectrum, ranges):
        """
        Normalize spectrum to [0, 1] and ranges by MAX_DA.
        """
        spectrum = np.asarray(spectrum)

        denom = spectrum.max() - spectrum.min()
        if denom == 0:
            spectrum_norm = np.zeros_like(spectrum, dtype=np.float32)
        else:
            spectrum_norm = (spectrum - spectrum.min()) / denom

        return spectrum_norm, ranges / MAX_DA

    def file2h5(self):
        spectrum_voxels = self.load_voxel_spectrum()
        rrs, ions = self.load_ranging()

        spectrum_o_all = []
        spectrum_log_all = []
        range_o_all = []

        for v in range(spectrum_voxels.shape[-1]):
            da_insert = np.random.random_sample((self.expand_factor,))
            top = min(307 - rrs.max(), self.shift_range[1])
            da_insert = da_insert * (top - self.shift_range[0]) + self.shift_range[0]

            for k, it in enumerate(da_insert):
                spectrum_o, range_o = self.apply_peakshift(
                    spectrum_voxels[:, v],
                    rrs,
                    da_insert=it,
                )

                spectrum_log = np.log(spectrum_o + 1)

                if self.norm:
                    spectrum_log, range_o = self.normalize(spectrum_log, range_o)

                spectrum_o_all.append(spectrum_o)
                spectrum_log_all.append(spectrum_log)
                range_o_all.append(range_o)

        spectrum_o_all = np.stack(spectrum_o_all).astype("float32")
        spectrum_log_all = np.stack(spectrum_log_all).astype("float32")
        range_o_all = np.stack(range_o_all).astype("float32")

        print("Augmentation finished for", Path(self.apt_file).stem)

        outdir = Path(self.savepath)
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"{Path(self.apt_file).stem}.h5"

        ion_data = np.array([str(x) for x in ions], dtype=object)

        with h5py.File(outfile, "w") as f:
            str_dt = h5py.string_dtype(encoding="utf-8")

            f.create_dataset("ion", data=ion_data, dtype=str_dt)
            f.create_dataset("if_norm", data=np.array(self.norm, dtype=np.bool_))
            f.create_dataset("if_peak_shift", data=np.array(True, dtype=np.bool_))
            f.create_dataset("input", data=spectrum_log_all)
            f.create_dataset("label", data=range_o_all)
            f.create_dataset("non_log_spectrums", data=spectrum_o_all)

        print("h5 file writing finished:", outfile)

        return outfile



'''
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
        try:
            d = apav.load_apt(self.apt_file)
        except:
            d = apav.load_pos(self.apt_file)
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
'''
