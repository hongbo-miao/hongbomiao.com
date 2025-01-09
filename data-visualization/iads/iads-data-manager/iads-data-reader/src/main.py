import logging
from pathlib import Path

from utils.iads_util import IadsUtil

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    iads_data_path = Path(r"data\Experiment1")
    iads_manager_exe_path = Path(
        r"C:\Program Files\IADS\DataManager\IadsDataManager.exe",
    )
    timezone = "UTC"
    df = IadsUtil.get_iads_dataframe(iads_manager_exe_path, iads_data_path, timezone)
    logger.info(f"{df = }")
