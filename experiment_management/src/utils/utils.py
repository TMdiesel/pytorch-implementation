import pathlib
import logging


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