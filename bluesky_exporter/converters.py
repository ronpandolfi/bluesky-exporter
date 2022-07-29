import itertools
import re
import shutil
import subprocess
import tempfile
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path
import time

import h5py
import numpy as np
from astropy.io import fits
from xarray import Dataset
from databroker.core import BlueskyRun
import tifffile
from xicam.core.data.bluesky_utils import streams_from_run
from pyqtgraph.parametertree import parameterTypes as ptypes

from databroker import Broker
from xicam.SAXS.operations.correction import correct
from xicam.core.threads import invoke_in_main_thread, invoke_as_event

from bluesky_exporter.dialogs import ParameterDialog

db = Broker.named('local').v2

tmp_dir = tempfile.tempdir


class Converter:
    name: str = None

    converter_classes = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.converter_classes[cls.name] = cls

    def __init__(self, export_dir: str):
        self.export_dir = export_dir

    def convert_run(self, run: BlueskyRun):
        sample_name = self.get_sample_name(run)
        for stream_name in streams_from_run(run):
            stream = getattr(run, stream_name)
            yield from self.convert_stream(stream, sample_name)

    def convert_stream(self, stream, sample_name):
        yield from self.convert_data(stream.to_dask(), sample_name, stream.name)

    def convert_data(self, data: Dataset, sample_name: str, stream_name: str) -> str:
        ...

    def get_sample_name(self, run):
        return run.metadata['start'].get('sample_name')


class NoOpConverter(Converter):
    name = 'None (export original data files)'

    def convert_run(self, run: BlueskyRun):
        sample_name = None
        resource_counter = itertools.count()

        for name, doc in run.canonical(fill='no'):
            if name == 'start':
                sample_name = slugify(doc['sample_name'])

            elif name == 'resource':
                for src_path in run.get_file_list(doc):
                    dest_path = (Path(self.export_dir) / Path(f"{sample_name}_{next(resource_counter)}")).with_suffix(Path(src_path).suffix)
                    shutil.copy2(src_path, dest_path)
        yield


class TiffConverter(Converter):
    name = 'Tiff'

    def convert_data(self, data: Dataset, sample_name: str, stream_name: str):
        for field_name in data.variables.keys():
            array = getattr(data, field_name).squeeze()
            if len(array.dims) in [2, 3]:
                dest_path = (Path(self.export_dir) / Path(f"{sample_name}_{stream_name}_{field_name}")).with_suffix('.tif')
                tifffile.imwrite(dest_path, array)
                yield


class FitsConverter(Converter):
    name = 'FITS'

    def convert_data(self, data: Dataset, sample_name: str, stream_name: str):
        for field_name in data.variables.keys():
            array = getattr(data, field_name).squeeze()
            if len(array.dims) in [2, 3]:
                dest_path = (Path(self.export_dir) / Path(f"{sample_name}_{stream_name}_{field_name}")).with_suffix('.fits')
                fits.PrimaryHDU(array).writeto(dest_path)
                yield


class CXIConverter(Converter):
    name = 'Cosmic CXI'
    
    def __init__(self, *args, **kwargs):
        super(CXIConverter, self).__init__(*args, **kwargs)

        self.dialog = None
        self.ready = False

        if int(subprocess.check_output('ulimit -n', shell=True).decode()) <= 1024:
            warnings.warn('The current value of ulimit is low. If you try to export very long runs, you may encounter '
                          'errors. It is recommended to increase this value with "ulimit -n".')

        def run_dialog():
            self.dialog = ParameterDialog(
                [bin_param := ptypes.SimpleParameter(name='Bin Multiple Exposures', value=True, type='bool'),
                 dark_param := ptypes.SimpleParameter(name='UID of Darks', value='', type='str'),
                 override_param := ptypes.SimpleParameter(name='Override Energy', value=False, type='bool'),
                 energy_param := ptypes.SimpleParameter(name='Energy', value=0, type='float', suffix='eV', siPrefix=True, visible=False),
                 ],
                'Please choose parameters for the following export options.')

            override_param.sigValueChanged.connect(lambda parameter, value: energy_param.setOpts(visible=value))

            self.dialog.open()
            self.dialog.accepted.connect(self._accepted)

        invoke_as_event(run_dialog)

    def _rejected(self):
        raise InterruptedError('Cancelled export from dialog.')

    def _accepted(self):
        parameter = self.dialog.get_parameters()
        self.override_energy = parameter['Override Energy']
        self.bin_frames = parameter['Bin Multiple Exposures']
        self.dark_uid = parameter['UID of Darks']
        self.energy = parameter['Energy']
        self.ready = True

    def convert_run(self, run: BlueskyRun):
        primary_stream = run.primary.to_dask()
        dark_stream = None
        if hasattr(run, 'darks'):
            darks_type = 'interleaved'
            dark_stream = run.darks.to_dask()
            print('Acquisition has a darks stream, using that for dark subtraction')
            warnings.warn('Dark stream subtraction using darks collected alongside the primary dataset is not yet implemented. I have no idea what will happen.')
        elif self.dark_uid != '':
            darks_type = 'single'
            darks_run = db[self.dark_uid.lower().strip()]
            if hasattr(darks_run, 'dark'):
                print('dark run has a dark stream, using that')
                dark_stream = darks_run.dark.to_dask()
            elif hasattr(darks_run, 'darks'):
                print('dark run has a "darks" stream, using that')
                dark_stream = darks_run.darks.to_dask()
            else:
                print('Dark run has no darks, using it\'s primary stream')
                dark_stream = darks_run.primary.to_dask()
        else:
            warnings.warn('No dark stream in this acquisition, and no dark acquisition was manually noted. Dark subtraction will be skipped')

        start_time = run.metadata['start']['time']
        end_time = None
        try:
            end_time = run.metadata['stop']['time']  # TODO: handle no stop doc
        except KeyError:
            warnings.warn('No stop document in run. Likely a failed/aborted acquisition.')

        # Extract the overall metadata
        # This gets the time into the correct format, ISO 8601. Perhaps there
        # is a native way to do this with the Timestamp object but I don't
        # know where to find the docs for it.
        times = [start_time]
        if end_time: times.append(end_time)
        times = tuple(datetime.fromtimestamp(time) for time in times)
        try:
            start_time, end_time = tuple(time.strftime('%Y-%m-%dT%H:%M:%S.%f') for time in times)
        except ValueError as e:
            # If there was an error and no recorded end time
            start_time, = tuple(time.strftime('%Y-%m-%dT%H:%M:%S.%f') for time in times)

        energy = np.mean(primary_stream['mono_energy'].compute())

        if self.override_energy:
            energy = self.energy

        energy = energy * 1.60218e-19  # to J
        wavelength = 1.9864459e-25 / energy

        x_locs = primary_stream['sample_translate'].compute() * 1e-6 
        y_locs = primary_stream['sample_lift'].compute() * 1e-6  
        # TODO: sample z motor?

        # Now package the translations into a format digestible by CXI
        # The negative sign makes the translations match up with the
        # detector's orientation.
        translations = np.stack((x_locs, -y_locs, np.zeros(x_locs.shape))).transpose()

        mask_run = db['f2d9e'] # TODO: this probably shouldn't be hardcoded
        mask = np.squeeze(np.asarray(mask_run.primary.to_dask()['fastccd_image']) == 8191)

        x_pixel_size, y_pixel_size = 30e-6, 30e-6  # m, pixel sizes
        distance = 0.34  

        delta = np.mean(np.asarray(primary_stream['detector_rotate']))
        gamma = 0  # TODO: pull from motors? np.asarray(primary_stream['det_translate'])
        basis_vectors = np.array([[0, -y_pixel_size, 0], [-x_pixel_size, 0, 0]]).transpose()
        # This should be redone
        corner_position = np.dot(basis_vectors, -np.array([980//2, 960//2]))  # TODO: confirm x size
        corner_position += np.array([0, 0, distance])
        # The patterns of negative signs below are to emphasize the correct
        # rotation matrices and the direction (clockwise or counterclockwise)
        # that rotation is defined in for the various diffractometer directions
        rotation_delta = np.array([[1, 0, 0],
                                   [0, np.cos(np.deg2rad(-delta)), -np.sin(np.deg2rad(-delta))],
                                   [0, np.sin(np.deg2rad(-delta)), np.cos(np.deg2rad(-delta))]])

        rotation_gamma = np.array([[np.cos(np.deg2rad(gamma)), 0, np.sin(np.deg2rad(gamma))],
                                   [0, 1, 0],
                                   [-np.sin(np.deg2rad(gamma)), 0, np.cos(np.deg2rad(gamma))]])

        rotation = np.dot(rotation_delta, rotation_gamma)
        basis_vectors = np.dot(rotation, basis_vectors)
        corner_position = np.dot(rotation, corner_position)
        # apply rotation matrix to basis vectors and corner position

        uid = run.metadata['start']['uid']

        # Create the data file
        path = str(Path(self.export_dir) / Path(f"{run.metadata['start']['scan_id']}_{uid}").with_suffix('.cxi'))

        with h5py.File(path, 'w') as f:
            f.create_dataset('cxi_version', data=150)
            f.create_dataset('number_of_entries', data=1)

            # Populate the major metadata fields
            entry_1 = f.create_group('entry_1')
            entry_1['start_time'] = np.string_(start_time)
            try:
                entry_1['end_time'] = np.string_(end_time)
            except UnboundLocalError:
                pass
            entry_1.create_dataset('run_id', data=uid)

            # Populate the sample and geometry
            sample_1 = entry_1.create_group('sample_1')
            sample_1['name'] = np.string_(run.metadata['start']['sample_name'])
            geometry_1 = sample_1.create_group('geometry_1')
            geometry_1.create_dataset('translation', data=translations)  # in m
            # geometry_1.create_dataset('surface_normal', data=surface_normal)  # TODO: revisit for sample rotation mode

            # Populate the detector and source information
            instrument_1 = entry_1.create_group('instrument_1')
            instrument_1['name'] = np.string_('COSMIC-Scattering')

            detector_1 = instrument_1.create_group('detector_1')
            detector_1.create_dataset('corner_position', data=corner_position)
            detector_1.create_dataset('delta', data=delta)
            detector_1.create_dataset('gamma', data=gamma)
            detector_1.create_dataset('x_pixel_size', data=x_pixel_size)
            detector_1.create_dataset('y_pixel_size', data=y_pixel_size)
            detector_1.create_dataset('basis_vectors', data=basis_vectors)
            detector_1.create_dataset('distance', data=distance)
            detector_1['translation'] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')

            source_1 = instrument_1.create_group('source_1')
            source_1['name'] = np.string_('ALS')
            source_1.create_dataset('energy', data=energy)  # in J
            source_1.create_dataset('wavelength', data=wavelength)  # in m

            # And finally, we create the data group
            data_1 = entry_1.create_group('data_1')
            data_1['translation'] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')

            images = primary_stream['fastccd_image']
            if dark_stream is not None:
                darks = dark_stream['fastccd_image']
                # In this case, we make a single dark frame to subtract
                # all the data, using a mean of all the dark frames we have
                if darks_type == 'single':
                    axes = tuple(range(len(darks.shape) - 2))
                    darks = np.mean(darks, axes).astype(np.float32)
                # TODO: implement a proper way to deal with the case where
                # dark frames are taken interspersed with the data.

            # squeeze extra dims
            if len(images.dims) > 4:
                raise ValueError('The scan appears to have too high of a dimension to be properly understood as ptychography')
            elif len(images.dims) == 4:
                if self.bin_frames:
                    # We compress multiple frames into one
                    images = np.mean(images,axis=-3)
                else:
                    # What needs to happen here is the translations will need
                    # to be expanded so each frame gets it's own translation
                    raise NotImplementedError('Support for not binning frames is not yet implemented')

            # We just always convert to float to avoid weird edge cases with
            # subtracting the background, which is usually a mean even when
            # the images themselves are from single frames (and thus still ints)
            images = images.astype(np.float32)

            # This defines the shape for the final data, which will have the
            # overscan slice excluded.
            overscan_slice = slice(966, 1084)
            data_shape = images.shape[:-1] + (images.shape[-1] - (1084-966),)

            # We save the mask here so we can get the right shape
            detector_1.create_dataset(
                'mask',
                data=np.delete(mask, overscan_slice, -1))
            
            # initialize dataset
            det1 = detector_1.create_dataset('data', shape=data_shape,
                                             dtype=np.float32)
            data_1['data'] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')
            # Currently we do no flat-field correction
            flats = np.ones_like(images[0])

            
            for i, frame in enumerate(images):
                frame = np.asarray(frame)

                if dark_stream is not None:
                    frame = correct(np.expand_dims(frame, 0), flats, darks)[0]

                frame = np.delete(frame, overscan_slice, -1)
                det1[i] = np.asarray(frame)

                yield i, len(images)

        # yield out all artifact paths (not actually used yet, WIP)
        yield path


class NxsasConverter(Converter):
    name = 'Nexus NXsas (Cosmic-Scattering)'

    def __init__(self, *args, **kwargs):
        super(NxsasConverter, self).__init__(*args, **kwargs)

        self.dialog = None
        self.ready = False

        def run_dialog():
            self.dialog = ParameterDialog(
                [ptypes.SimpleParameter(name='X Min', value=0, type='int'),
                 ptypes.SimpleParameter(name='X Max', value=-1, type='int'),
                 ptypes.SimpleParameter(name='Y Min', value=0, type='int'),
                 ptypes.SimpleParameter(name='Y Max', value=-1, type='int'),
                 ],
                "Enter the export ROI ranges (optional). A max of -1 means 'end'.")
            self.dialog.open()
            self.dialog.accepted.connect(self._accepted)

        invoke_as_event(run_dialog)

    def _rejected(self):
        raise InterruptedError('Cancelled export from dialog.')

    def _accepted(self):
        self.x_min = self.dialog.get_parameters()['X Min']
        self.y_min = self.dialog.get_parameters()['Y Min']
        self.x_max = self.dialog.get_parameters()['X Max']
        self.y_max = self.dialog.get_parameters()['Y Max']
        self.ready = True

    def convert_run(self, run: BlueskyRun):
        from xicam.SAXS.operations.correction import correct

        # statics for fastccd
        distance = 0.284
        x_pixel_size = y_pixel_size = 30e-6

        uid = run.metadata['start']['uid']

        # Create the data file
        path = Path(self.export_dir) / Path(f"{run.metadata['start']['sample_name']}_{uid}")
        path = str(path.with_suffix(path.suffix+'.h5'))

        primary_stream = run.primary.to_dask()
        labview_stream = run.labview.to_dask()

        with h5py.File(path, 'w') as f:

            raw = primary_stream['fastccd_image']
            dark = np.average(np.squeeze(run.dark.to_dask()['fastccd_image']), axis=0)
            flats = np.ones(raw.shape[-2:])

            # when ndim > 3, squeeze extra dims
            dims_to_squeeze = len(raw.dims)-3
            if dims_to_squeeze > 0:
                for i in range(len(raw.dims)):
                    if raw.shape[i] == 1:
                        raw = np.squeeze(raw, axis=i)
                        dims_to_squeeze -= 1
                        if dims_to_squeeze == 0:
                            break

            x_max = raw.shape[-1] if self.x_max == -1 else self.x_max
            y_max = raw.shape[-2] if self.y_max == -1 else self.y_max

            dark = dark[self.y_min:y_max+1, self.x_min:x_max+1]
            flats = flats[self.y_min:y_max+1, self.x_min:x_max+1]

            start_time = run.metadata['start']['time']
            end_time = None
            try:
                end_time = run.metadata['stop']['time']  # TODO: handle no stop doc
            except KeyError:
                warnings.warn('No stop document in run. Likely a failed/aborted acquisition.')

            # Extract the overall metadata
            # This gets the time into the correct format, ISO 8601. Perhaps there is a native way to
            # do this with the Timestamp object but I don't know where to find the docs for it.
            times = [start_time]
            if end_time: times.append(end_time)
            times = tuple(datetime.fromtimestamp(time) for time in times)
            try:
                start_time, end_time = tuple(time.strftime('%Y-%m-%dT%H:%M:%S.%f') for time in times)
            except ValueError as e:
                # If there was an error and no recorded end time
                start_time, = tuple(time.strftime('%Y-%m-%dT%H:%M:%S.%f') for time in times)

            energy = np.mean(labview_stream['mono_energy'].compute())
            energy = energy * 1.60218e-19  # to J
            wavelength = 1.9864459e-25 / energy

            # Populate the major metadata fields
            entry_1 = f.create_group('entry1')
            entry_1['start_time'] = np.string_(start_time)
            try:
                entry_1['end_time'] = np.string_(end_time)
            except UnboundLocalError:
                pass
            entry_1.create_dataset('run_id', data=uid)

            # Populate the sample and geometry
            sample_1 = entry_1.create_group('sample_1')
            sample_1['name'] = np.string_(run.metadata['start']['sample_name'])
            geometry_1 = sample_1.create_group('geometry_1')
            # geometry_1.create_dataset('surface_normal', data=surface_normal)  # TODO: revisit for sample rotation mode

            # Populate the detector and source information
            instrument_1 = entry_1.create_group('instrument_1')
            instrument_1['name'] = np.string_('COSMIC-Scattering')

            detector_1 = instrument_1.create_group('detector_1')
            detector_1.create_dataset('description', data='LBNL FastCCD')
            detector_1.create_dataset('distance', data=distance)
            detector_1.create_dataset('x_pixel_size', data=x_pixel_size)
            detector_1.create_dataset('y_pixel_size', data=y_pixel_size)
            detector_1.create_dataset('count_time', data=run.primary.metadata['descriptors'][0]['configuration']['fastccd']['data']['fastccd_cam_acquire_time'])
            detector_1.create_dataset('period', data=run.primary.metadata['descriptors'][0]['configuration']['fastccd']['data']['fastccd_cam_acquire_period'])
            detector_1.create_dataset('exposures', data=run.primary.metadata['descriptors'][0]['configuration']['fastccd']['data']['fastccd_cam_num_exposures'])

            det1 = detector_1.create_dataset('data', shape=(raw.shape[0], y_max-self.y_min, x_max-self.x_min))

            for i, raw_frame in enumerate(raw):
                raw_frame = np.asarray(raw_frame[self.y_min:y_max, self.x_min:x_max])
                corrected_image = correct(np.expand_dims(raw_frame, 0), flats, dark)[0]

                det1[i] = corrected_image
                yield i, len(raw)

            # Add LabVIEW data
            labview_group = instrument_1.create_group('labview_data')
            for field in labview_stream:
                labview_group.create_dataset(field, data=labview_stream[field].compute())

        yield path


class Intake(Converter):
    name = 'Intake'

    def convert_run(self, run: BlueskyRun):
        sample_name = self.get_sample_name(run)
        dest_path = (Path(self.export_dir) / Path(f"{sample_name}")).with_suffix('.yaml')
        run.export(dest_path)
        yield


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')



