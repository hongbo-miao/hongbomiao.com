import logging

import can
import cantools


def main() -> None:
    db = cantools.database.load_file("src/dbc/engine.dbc")
    logging.info(db.messages)
    example_message = db.get_message_by_name("EEC1")
    logging.info(example_message.signals)

    data = example_message.encode({"EngineSpeed": 200.1})
    message = can.Message(arbitration_id=example_message.frame_id, data=data)
    logging.info(message)

    can_bus = can.interface.Bus("vcan0", bustype="socketcan")
    can_bus.send(message)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
