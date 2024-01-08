import logging

import pyvisa

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    resource_manager = pyvisa.ResourceManager()
    logging.info(resource_manager.list_resources())
    logging.info(resource_manager.list_opened_resources())

    instrument = resource_manager.open_resource("ASRL1::INSTR")
    logging.info(instrument)

    instrument.close()
