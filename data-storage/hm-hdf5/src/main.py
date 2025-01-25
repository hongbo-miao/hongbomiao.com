import logging
from pathlib import Path

import h5py
import polars as pl

logger = logging.getLogger(__name__)


class Hdf5Util:
    @staticmethod
    def process_file(
        hdf5_path: Path,
    ) -> dict[str, pl.DataFrame]:
        try:
            with h5py.File(hdf5_path, "r") as file:
                dfs: dict[str, pl.DataFrame] = {}

                def traverse_h5(name: str, obj: h5py.Dataset | h5py.Group) -> None:
                    if isinstance(obj, h5py.Dataset):
                        data = obj[:]
                        logger.info(f"{name = }, {data.shape = }")

                        if len(data.shape) == 2:
                            if name not in dfs:
                                dfs[name] = pl.DataFrame(data)
                        else:
                            logger.warning(
                                f"{len(data.shape)}-dimensional. Skipping.",
                            )

                file.visititems(traverse_h5)
                return dfs

        except Exception:
            logger.exception("Error reading HDF5 file")
            return {}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    data_dir_path = Path("data")
    hdf5_path = data_dir_path / Path("data.h5")
    dfs = Hdf5Util.process_file(hdf5_path)
    for name, df in dfs.items():
        logger.info(f"{name = } {df = }")
