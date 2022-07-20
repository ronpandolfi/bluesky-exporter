import warnings

from bluesky_live import conversion


def _transpose(in_data, keys, data_keys, field):
    """Turn a list of dicts into dict of lists

    Parameters
    ----------
    in_data : list
        A list of dicts which contain at least one dict.
        All of the inner dicts must have at least the keys
        in `keys`

    keys : list
        The list of keys to extract

    field : str
        The field in the outer dict to use

    Returns
    -------
    transpose : dict
        The transpose of the data
    """
    import dask.array
    import numpy

    out = {k: [None] * len(in_data) for k in keys}
    for j, ev in enumerate(in_data):
        dd = ev[field]
        for k in keys:
            out[k][j] = dd[k]
    for k in keys:
        try:
            if len(out[k]):
                out[k] = dask.array.stack(out[k])
            else:
                # Case of no Events yet
                out[k] = dask.array.array([]).reshape(0, *data_keys[k]["shape"])
        except ValueError as ex:  # Monkey patch case to handle malformed fastccd data; obliterates the first data point in an acquisition
            if out[k][0].shape[0] != out[k][1].shape[0]:
                out[k][0] = out[k][1]
                out[k] = dask.array.stack(out[k])
                warnings.warn('ATTENTION! The selected run had malformed data as a result of an acquisition bug. In '
                              'order to accommodate export, the first data point in the sequence will be made invalid. '
                              'Please consider this in further analysis.')
            else:
                raise ValueError('Error sustained.') from ex
            
        except NotImplementedError:
            # There are data structured that dask auto-chunking cannot handle,
            # such as an list of list of variable length. For now, let these go
            # out as plain numpy arrays. In the future we might make them dask
            # arrays with manual chunks.
            out[k] = numpy.asarray(out[k])

    return out


# inject monkey patch
conversion.__dict__['_transpose'] = _transpose
