import itertools
import re
import shutil
import tempfile
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from pyqtgraph.parametertree import parameterTypes as ptypes
from astropy.io import fits
from xarray import Dataset
from databroker.core import BlueskyRun
import tifffile
from xicam.core.data.bluesky_utils import streams_from_run
from databroker import Broker
from xicam.SAXS.operations.correction import correct

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

    def convert_run(self, run: BlueskyRun):
        primary_stream = run.primary.to_dask()
        dark_stream = None
        if hasattr(run, 'darks'):
            dark_stream = run.darks.to_dask()
        else:
            warnings.warn('No dark stream in this acquisition. Dark subtraction will be skipped')

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

        energy = np.mean(primary_stream['mono_energy'].compute())

        # NOTE: This is an example of how to display a dialog to the user asking to verify values
        dialog = ParameterDialog([ptypes.SimpleParameter(name='Energy', value=energy, type='float', suffix='eV', siPrefix=True)],
                                 'The values below are derived from the acquisition. You can override them here if desired.')
        if dialog.exec_():
            energy = dialog.get_parameters()['Energy']
        else:
            raise InterruptedError('Cancelled export from dialog.')

        energy = energy * 1.60218e-19  # to J
        wavelength = 1.9864459e-25 / energy

        x_locs = primary_stream['sample_translate'].compute() * 1e-3  # TODO: confirm motor name
        y_locs = primary_stream['sample_lift'].compute() * 1e-3  # TODO: confirm motor name
        # TODO: sample z motor?

        # Now package the translations into a format digestible by CXI
        translations = np.stack((x_locs, y_locs, np.zeros(x_locs.shape))).transpose()

        mask_run = db['f2d9e']
        mask = np.squeeze(np.asarray(mask_run.primary.to_dask()['fastccd_image']) == 8191)

        x_pixel_size, y_pixel_size = 30e-6, 30e-6  # m, pixel sizes
        distance = 0.34  # TODO: ask sophie/sujoy

        delta = np.mean(np.asarray(primary_stream['detector_rotate']))
        gamma = 0  # TODO: pull from motors? np.asarray(primary_stream['det_translate'])
        basis_vectors = np.array([[0, -y_pixel_size, 0], [-x_pixel_size, 0, 0]]).transpose()
        # This should be redone
        corner_position = np.dot(basis_vectors, -np.array([980//2, 960//2]))  # TODO: confirm x size
        corner_position += np.array([0, 0, distance])
        # The patterns of negative signs below are to emphasize the correct rotation matrices
        # and the direction (clockwise or counterclockwise) that rotation is defined in for the
        # various diffractometer directions
        rotation_delta = np.array([[1, 0, 0],
                                   [0, np.cos(np.deg2rad(-delta)), -np.sin(np.deg2rad(-delta))],
                                   [0, np.sin(np.deg2rad(-delta)), np.cos(np.deg2rad(-delta))]])
        # check if gamma should be positive or negative here
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

        # Note from Abe: I changed this because not closing the files
        # was causing it to leak memory over time
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
            detector_1.create_dataset('mask', data=mask)
            detector_1['translation'] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')

            source_1 = instrument_1.create_group('source_1')
            source_1['name'] = np.string_('ALS')
            source_1.create_dataset('energy', data=energy)  # in J
            source_1.create_dataset('wavelength', data=wavelength)  # in m

            # And finally, we create the data group
            data_1 = entry_1.create_group('data_1')
            data_1['translation'] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')

            images = primary_stream['fastccd_image']
            overscan_slice = slice(966, 1084)

            # decide how big to make dataset for images
            if dark_stream is not None:
                data_shape = np.delete(images, overscan_slice, -1).shape
            else:
                data_shape = images.shape

            # initialize dataset
            det1 = detector_1.create_dataset('data', shape=data_shape)

            # squeeze extra dims
            if len(images.dims) > 3:
                images = np.squeeze(images)

            flats = np.ones_like(images[0])
            for i, frame in enumerate(images):
                if dark_stream is not None:
                    # subtract darks
                    frame = correct(np.expand_dims(frame, 0), flats, dark_stream['fastccd_image'])[0]
                    # remove overscan
                    frame = np.delete(frame, overscan_slice, -1)
                    # TODO: rotate?
                    # write data
                det1[i] = np.asarray(frame)

        # yield out all artifact paths (not actually used yet, WIP)
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



