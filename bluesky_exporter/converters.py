import itertools
import re
import shutil
import tempfile
import unicodedata
from pathlib import Path

from astropy.io import fits
from xarray import Dataset
from databroker.core import BlueskyRun
from astropy.table import Table
import tifffile
from xicam.core.data.bluesky_utils import streams_from_run

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


class Intake(Converter):
    name = 'Intake'
    def convert_run(self, run: BlueskyRun):
        dest_path = (Path(self.export_dir) / Path(f"{sample_name}_{stream_name}_{field_name}")).with_suffix('.tif')
        run.export(dest_path)


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