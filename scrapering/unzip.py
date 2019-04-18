import zstandard
import pathlib
import shutil
import os


input_file = os.path.join("RC_2019-01.zst")
with open(input_file, 'rb') as compressed:
    decomp = zstandard.ZstdDecompressor()
    output_path = os.path.join("output.txt")
    with open(output_path, 'wb') as destination:
        decomp.copy_stream(compressed, destination)