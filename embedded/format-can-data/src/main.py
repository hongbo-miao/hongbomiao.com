import logging
from pathlib import Path

import can
import cantools


def main() -> None:
    dbc = cantools.database.load_file(Path("src/dbc/engine.dbc"))
    logging.info(dbc.messages)
    eec1_message_definition = dbc.get_message_by_name("EEC1")
    logging.info(eec1_message_definition.signals)

    data = eec1_message_definition.encode({"EngineSpeed": 200.1})
    frame = can.Message(arbitration_id=eec1_message_definition.frame_id, data=data)
    logging.info(frame)

    can_bus = can.interface.Bus("vcan0", bustype="socketcan")
    can_bus.send(frame)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
