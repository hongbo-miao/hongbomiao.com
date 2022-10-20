import time

import rti.connextdds as dds


def publisher_main(domain_id: int):
    participant = dds.DomainParticipant(domain_id=domain_id)
    hm_type = dds.QosProvider("hm_message.xml").type("HMMessage")
    sample = dds.DynamicData(hm_type)
    topic = dds.DynamicData.Topic(participant, "hm-topic", hm_type)
    writer = dds.DynamicData.DataWriter(dds.Publisher(participant), topic)

    count = 0
    while True:
        sample["count"] = count
        sample["name"] = "odd" if count % 2 == 1 else "even"
        print(sample)
        writer.write(sample)
        time.sleep(1)
        count += 1


if __name__ == "__main__":
    dds_domain_id = 0
    publisher_main(dds_domain_id)
