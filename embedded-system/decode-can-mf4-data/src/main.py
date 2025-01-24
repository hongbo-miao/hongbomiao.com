import logging
from pathlib import Path

import pandas as pd
from asammdf import MDF

logger = logging.getLogger(__name__)


class Mf4Util:
    @staticmethod
    def process_file(
        mf4_path: Path,
    ) -> pd.DataFrame:
        try:
            mdf = MDF(mf4_path)
            return mdf.to_dataframe()
        except Exception:
            logger.exception("Error processing MF4 file")
        finally:
            mdf.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    data_dir_path = Path("data")
    mf4_path = data_dir_path / Path("can.mf4")
    df = Mf4Util.process_file(mf4_path)
    logger.info(df)
