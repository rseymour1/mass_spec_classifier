#!pip install depthcharge-ms
#!pip install pylance==0.15

import run_depthcharge as dc
from depthcharge.encoders import PeakEncoder
from depthcharge.data import SpectrumDataset
import torch

mzml = './orbitrap_lumos/08CPTAC_C_GBM_W_PNNL_20210830_B2S3_f20.mzML'
df = dc.data.spectra_to_df(mzml, progress=True)

encoder = PeakEncoder(100)
dataset = SpectrumDataset(mzml, batch_size=1000)
count = 0
for spectrum in dataset:
    # embeded = encoder.forward(spectrum["mz_array"])
    a_spec = spectrum
    if (count == 0):
        break  
    count += 1

mz_values = a_spec["mz_array"][0]
intensities = a_spec["intensity_array"][0]
peaks = list(zip(mz_values, intensities))

test1 = torch.tensor([peaks])
encoded_test1 = encoder.forward(test1)

from torch.utils.data import DataLoader
from depthcharge.transformers import SpectrumTransformerEncoder

model = SpectrumTransformerEncoder()
# Grab the first 1000 spectra for testing
#b_size = 1000
for batch in dataset:
    out = model(batch["mz_array"], batch["intensity_array"])
    print(out[0].shape)
    break

