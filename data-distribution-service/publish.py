import logging
import time

import rti.connextdds as dds

logger = logging.getLogger(__name__)


def publisher_main(dds_domain_id: int, total_count: int) -> None:
    participant = dds.DomainParticipant(domain_id=dds_domain_id)
    hm_type = dds.QosProvider("hm_message.xml").type("HMMessage")

    topic = dds.DynamicData.Topic(participant, "hm-topic", hm_type)
    writer = dds.DynamicData.DataWriter(dds.Publisher(participant), topic)

    sample = dds.DynamicData(hm_type)
    handle = writer.register_instance(sample)

    count = 0
    while count < total_count:
        sample["count"] = count
        sample["name"] = "odd" if count % 2 == 1 else "even"
        logger.info(sample)
        writer.write(sample, handle)
        time.sleep(1)
        count += 1

    writer.unregister_instance(handle)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    external_dds_domain_id = 0
    external_total_count = 100
    publisher_main(external_dds_domain_id, external_total_count)
