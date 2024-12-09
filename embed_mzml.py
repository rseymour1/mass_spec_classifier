import polars as pl
import depthcharge as dc

mzml_file = ""

# Read an mzML into a DataFrame:
df = dc.data.spectra_to_df(mzml_file, progress=False)
print(df.head())

batch = next(dc.data.spectra_to_stream(mzml_file, progress=False))
print(pl.from_arrow(batch))