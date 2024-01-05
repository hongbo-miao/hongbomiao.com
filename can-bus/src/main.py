import logging

import can
import cantools


def main() -> None:
    db = cantools.database.load_file("src/dbc/engine.dbc")
    logging.info(db.messages)
    eec1_message = db.get_message_by_name("EEC1")
    logging.info(eec1_message.signals)

    data = eec1_message.encode({"EngineSpeed": 200.1})
    message = can.Message(arbitration_id=eec1_message.frame_id, data=data)
    logging.info(message)

    can_bus = can.interface.Bus("vcan0", bustype="socketcan")
    can_bus.send(message)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
