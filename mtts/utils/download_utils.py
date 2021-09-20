import hashlib
import os
import random
import sys
import urllib
import urllib.request
from typing import Any, Iterable, Optional

import torch
from tqdm import tqdm


def stream_url(url: str,
               start_byte: Optional[int] = None,
               block_size: int = 32 * 1024,
               progress_bar: bool = True) -> Iterable:
    """Stream url by chunk
    Args:
        url (str): Url.
        start_byte (int, optional): Start streaming at that point (Default: ``None``).
        block_size (int, optional): Size of chunks to stream (Default: ``32 * 1024``).
        progress_bar (bool, optional): Display a progress bar (Default: ``True``).
    """

    # If we already have the whole file, there is no need to download it again
    req = urllib.request.Request(url, method="HEAD")
    url_size = int(urllib.request.urlopen(req).info().get("Content-Length", -1))
    if url_size == start_byte:
        return

    req = urllib.request.Request(url)
    if start_byte:
        req.headers["Range"] = "bytes={}-".format(start_byte)

    with urllib.request.urlopen(req) as upointer, tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=url_size,
            disable=not progress_bar,
    ) as pbar:

        num_bytes = 0
        while True:
            chunk = upointer.read(block_size)
            if not chunk:
                break
            yield chunk
            num_bytes += len(chunk)
            pbar.update(len(chunk))


def validate_file(file_obj: Any, hash_value: str, hash_type: str = "sha256") -> bool:
    """Validate a given file object with its hash.
    Args:
        file_obj: File object to read from.
        hash_value (str): Hash for url.
        hash_type (str, optional): Hash type, among "sha256" and "md5" (Default: ``"sha256"``).
    Returns:
        bool: return True if its a valid file, else False.
    """

    if hash_type == "sha256":
        hash_func = hashlib.sha256()
    elif hash_type == "md5":
        hash_func = hashlib.md5()
    else:
        raise ValueError

    while True:
        # Read by chunk to avoid filling memory
        chunk = file_obj.read(1024**2)
        if not chunk:
            break
        hash_func.update(chunk)

    return hash_func.hexdigest() == hash_value


def download_url(url: str,
                 download_folder: str,
                 filename: Optional[str] = None,
                 hash_value: Optional[str] = None,
                 hash_type: str = "sha256",
                 progress_bar: bool = True,
                 resume: bool = False) -> None:
    """Download file to disk.
    Args:
        url (str): Url.
        download_folder (str): Folder to download file.
        filename (str, optional): Name of downloaded file. If None, it is inferred from the url (Default: ``None``).
        hash_value (str, optional): Hash for url (Default: ``None``).
        hash_type (str, optional): Hash type, among "sha256" and "md5" (Default: ``"sha256"``).
        progress_bar (bool, optional): Display a progress bar (Default: ``True``).
        resume (bool, optional): Enable resuming download (Default: ``False``).
    """

    req = urllib.request.Request(url, method="HEAD")
    req_info = urllib.request.urlopen(req).info()

    # Detect filename
    filename = filename or req_info.get_filename() or os.path.basename(url)
    filepath = os.path.join(download_folder, filename)
    if resume and os.path.exists(filepath):
        mode = "ab"
        local_size: Optional[int] = os.path.getsize(filepath)

    elif not resume and os.path.exists(filepath):
        raise RuntimeError("{} already exists. Delete the file manually and retry.".format(filepath))
    else:
        mode = "wb"
        local_size = None

    if hash_value and local_size == int(req_info.get("Content-Length", -1)):
        with open(filepath, "rb") as file_obj:
            if validate_file(file_obj, hash_value, hash_type):
                return
        raise RuntimeError("The hash of {} does not match. Delete the file manually and retry.".format(filepath))

    with open(filepath, mode) as fpointer:
        for chunk in stream_url(url, start_byte=local_size, progress_bar=progress_bar):
            fpointer.write(chunk)

    with open(filepath, "rb") as file_obj:
        if hash_value and not validate_file(file_obj, hash_value, hash_type):
            raise RuntimeError("The hash of {} does not match. Delete the file manually and retry.".format(filepath))


def download_checkpoint():
    url = 'https://zenodo.org/record/4625672/files/checkpoint_500000.pth'
    os.makedirs('./checkpoint/', exist_ok=True)
    return download_url(url,
                        './checkpoint/',
                        resume=True,
                        hash_value='14002c23879f6b5d0cd987f3c3e1a160',
                        hash_type='md5')


def download_waveglow(device):

    os.makedirs('./waveglow/', exist_ok=True)

    try:
        waveglow = torch.hub.load('./waveglow/DeepLearningExamples-torchhub/', 'nvidia_waveglow', source='local')
    except Exception:
        print((f'error occur: {sys.exc_info()}, If this occurs again, ' +
               'try to delete anyting in ./waveglow/DeepLearningExamples-torchhub/'))
        if random.randint(0, 1) == 0:
            download_url('https://hub.fastgit.org/nvidia/DeepLearningExamples/archive/torchhub.zip',
                         './waveglow',
                         hash_type='md5',
                         hash_value='27ef24b9c4a2ce6c26f26998aee26f44',
                         resume=True)
        else:
            download_url('https://github.com/nvidia/DeepLearningExamples/archive/torchhub.zip',
                         './waveglow',
                         hash_type='md5',
                         hash_value='27ef24b9c4a2ce6c26f26998aee26f44',
                         resume=True)
        os.system('unzip ./waveglow/DeepLearningExamples-torchhub.zip -d ./waveglow/')
        waveglow = torch.hub.load('./waveglow/DeepLearningExamples-torchhub/', 'nvidia_waveglow', source='local')

    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.eval()
    for m in waveglow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')
    waveglow.to(device)
    return waveglow
