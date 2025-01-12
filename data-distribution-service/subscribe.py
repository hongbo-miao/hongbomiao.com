import logging
import time

import rti.connextdds as dds

logger = logging.getLogger(__name__)


class Listener(dds.DynamicData.NoOpDataReaderListener):
    def on_data_available(self, reader: dds.DynamicData.DataReader) -> None:
        with reader.take() as samples:
            for sample in samples:
                if sample.info.valid:
                    logger.info(sample.data)


def subscriber_main(domain_id: int) -> None:
    participant = dds.DomainParticipant(domain_id)

    hm_type = dds.QosProvider("hm_message.xml").type("HMMessage")
    topic = dds.DynamicData.Topic(participant, "hm-topic", hm_type)
    reader = dds.DynamicData.DataReader(dds.Subscriber(participant), topic)
    listener = Listener()
    reader.bind_listener(listener, dds.StatusMask.DATA_AVAILABLE)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    dds_domain_id = 0
    subscriber_main(dds_domain_id)
