import os
import subprocess
import tarfile
from pathlib import Path
from typing import List

import requests
from tqdm import tqdm

core_file_name: str = "2019-07-01-ASR-public_12020.tar"
core_download_url: str = f"https://zenodo.org/record/3370144/files/{core_file_name}?download=1"
core_file_size: int = 263168000  # Bytes

# core_download_csv_url = "https://zenodo.org/record/3370144/files/2019-07-01-ASR-public_12020.csv?download=1"


input_path = Path("inputs")
output_path = Path("outputs")


def download_core():
    def download_file(file_url, output_file_path):
        response = requests.get(file_url, stream=True)
        total_bytes = int(response.headers.get('Content-Length'))
        progress_bar = tqdm(total=total_bytes, unit='iB', unit_scale=True, ncols=80)
        with open(output_file_path, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):  # 1 KiB
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if progress_bar.n != total_bytes:
            print("Downloaded file size mismatch!")
            exit(1)

    download_path = Path('core')
    download_path.mkdir(exist_ok=True)

    local_file = download_path / core_file_name
    if not local_file.exists():
        download_file(core_download_url, str(local_file))

    if os.path.getsize(local_file) != core_file_size:
        print("CoRE dataset size mismatch!")
        exit(1)

    if not (download_path / 'structure_11660').exists():
        with tarfile.open(local_file) as core_tar:
            core_tar.extractall(download_path)


def run(cmd: List[str], debug=False):
    if debug:
        print(f"Executing: {' '.join(cmd)}")

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # signal.alarm(1000)

    (stdout, stderr) = (x.decode('utf-8').strip() for x in p.communicate())
    if p.returncode != 0:
        print(f"ERROR CODE: {p.returncode}")
    if stderr:
        print(stderr)
        exit(1)

    if stdout:
        print(stdout)


def generate_inputs():
    input_path.mkdir(exist_ok=True)

    for p in Path("core/structure_11660").iterdir():
        print(f"Processing: {p}")
        name = p.name[:p.name.rindex('.')]
        run(['./bin/cif2input', str(p), 'MOFGAN/data_ff_UFF', str(input_path / f"{name}.input")])


def generate_energy_grids():
    output_path.mkdir(exist_ok=True)

    for p in input_path.iterdir():
        print(f"Processing: {p}")
        name = p.name[:p.name.rindex('.')]
        run(['./bin/Vext_cpu', str(p), str(output_path / f"{name}.output")])


def main():
    # Note: First clone https://github.com/MusenZhou/MOFGAN into this directory and compile
    download_core()
    # generate_inputs()
    # generate_energy_grids()

    print("DONE!")


if __name__ == '__main__':
    main()
