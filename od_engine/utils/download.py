from pathlib import Path

from torch.hub import download_url_to_file
import zipfile
from concurrent.futures import ThreadPoolExecutor


def download(url, dir=".", unzip=True, delete=True, threads=1):
    """
    Downloads and unzips files from a URL or list of URLs using PyTorch Hub.

    Args:
        url (str or list): The URL or list of URLs to download from.
        dir (str, optional): The directory to save the files in. Defaults to ".".
        unzip (bool, optional): Whether to unzip the downloaded files. Defaults to True.
        delete (bool, optional): Whether to delete the zip file after unzipping. Defaults to True.
        threads (int, optional): The number of threads to use for parallel downloads. Defaults to 1.
    """

    def _download_one(url, dir, unzip, delete):
        """Helper function to download and process a single URL."""
        f = Path(dir) / Path(url).name  # Filename
        dir = f.parent
        dir.mkdir(parents=True, exist_ok=True)  # Create dir if it does not exist

        print(f"Downloading {url} to {f}...")
        download_url_to_file(url, str(f), progress=True)

        if unzip and f.suffix == ".zip":
            print(f"Unzipping {f}...")
            with zipfile.ZipFile(f, "r") as zip_ref:
                zip_ref.extractall(path=dir)  # Unzip to dir
            if delete:
                f.unlink()  # Delete zip file

    urls = [url] if isinstance(url, str) else url
    if threads > 1 and len(urls) > 1:
        with ThreadPoolExecutor(threads) as executor:
            executor.map(
                _download_one,
                urls,
                [dir] * len(urls),
                [unzip] * len(urls),
                [delete] * len(urls),
            )
    else:
        for u in urls:
            _download_one(u, dir, unzip, delete)
