import polars as pl
import depthcharge as dc

mzml_file = ""

# Read an mzML into a DataFrame:
df = dc.data.spectra_to_df(mzml_file, progress=False)
print(df.head())

batch = next(dc.data.spectra_to_stream(mzml_file, progress=False))
print(pl.from_arrow(batch))
processing_fn = [
    #https://spectrum-utils.readthedocs.io/en/latest/api.html#spectrum_utils.spectrum.MsmsSpectrum.set_mz_range
    dc.data.preprocessing.set_mz_range(min_mz=0),
    #https://spectrum-utils.readthedocs.io/en/latest/api.html#spectrum_utils.spectrum.MsmsSpectrum.filter_intensity
    dc.data.preprocessing.filter_intensity(min_intensity = 0.0),
    #https://spectrum-utils.readthedocs.io/en/latest/api.html#spectrum_utils.spectrum.MsmsSpectrum.scale_intensity
    dc.data.preprocessing.scale_intensity(scaling=None),  # Might want to change later
    dc.data.preprocessing.scale_to_unit_norm,
]
df = dc.data.spectra_to_df(
    mzml_file,
    progress=False,
    preprocessing_fn=processing_fn
)
print(df.head())