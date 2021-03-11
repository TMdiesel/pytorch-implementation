import pathlib
import logging
from contextlib import contextmanager
import time


def init_root_logger(
    outdir: pathlib.Path,
    filename_normal:str="log.log",
    filename_error:str="error.log",
    ):

    outdir.mkdir(exist_ok=True)
    logging.getLogger().addHandler(
        _add_handler(outdir,logging.INFO,filename_normal))
    logging.getLogger().addHandler(
        _add_handler(outdir,logging.ERROR,filename_error))


def _add_handler(
    outdir:pathlib.Path,
    level,
    filename
    ):

    fh = logging.FileHandler(outdir/filename)
    fh.setLevel(level)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s')
    fh.setFormatter(fh_formatter)
    return fh


@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
