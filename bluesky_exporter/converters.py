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
from astropy.io import fits
from xarray import Dataset
from databroker.core import BlueskyRun
from astropy.table import Table
import tifffile
from xicam.core.data.bluesky_utils import streams_from_run
from pyqtgraph.parametertree import parameterTypes as ptypes

from bluesky_exporter.dialogs import ParameterDialog

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
        from suitcase.cxi import export

        # get the documents generator from the run object
        docs = run.documents(fill='yes')

        # run cxi exporter on the document stream
        artifacts = export(docs, self.export_dir)

        # yield out all artifact paths (not actually used yet, WIP)
        yield from sum(list(artifacts.values()))


class NxsasConverter(Converter):
    name = 'Nexus NXsas (Cosmic-Scattering)'

    def convert_run(self, run: BlueskyRun):
        from xicam.SAXS.operations.correction import correct

        # statics for fastccd
        distance = 0.284
        x_pixel_size = y_pixel_size = 30e-6

        uid = run.metadata['start']['uid']

        # Create the data file
        path = str(Path(self.export_dir) / Path(f"{run.metadata['start']['sample_name']}_{uid}").with_suffix('.h5'))

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

            dialog = ParameterDialog(
                [ptypes.SimpleParameter(name='X Min', value=0, type='int', limits=(0, flats.shape[-1])),
                 ptypes.SimpleParameter(name='X Max', value=flats.shape[-1], type='int', limits=(0, flats.shape[-1])),
                 ptypes.SimpleParameter(name='Y Min', value=0, type='int', limits=(0, flats.shape[-2])),
                 ptypes.SimpleParameter(name='Y Max', value=flats.shape[-2], type='int', limits=(0, flats.shape[-1])),
                 ],
                'Enter the export ROI ranges (optional).')

            if not dialog.exec_():
                raise InterruptedError('Cancelled export from dialog.')

            x_min = dialog.get_parameters()['X Min']
            y_min = dialog.get_parameters()['Y Min']
            x_max = dialog.get_parameters()['X Max']
            y_max = dialog.get_parameters()['Y Max']

            dark = dark[y_min:y_max+1, x_min:x_max+1]
            flats = flats[y_min:y_max+1, x_min:x_max+1]

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

            det1 = detector_1.create_dataset('data', shape=(raw.shape[0], y_max-y_min, x_max-x_min))

            for i, raw_frame in enumerate(raw):
                raw_frame = np.asarray(raw_frame[y_min:y_max, x_min:x_max])
                corrected_image = correct(np.expand_dims(raw_frame, 0), flats, dark)[0]

                det1[i] = corrected_image

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