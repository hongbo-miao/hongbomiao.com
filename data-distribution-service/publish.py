import time

import rti.connextdds as dds


def publisher_main(domain_id: int, total_count: int):
    participant = dds.DomainParticipant(domain_id=domain_id)
    hm_type = dds.QosProvider("hm_message.xml").type("HMMessage")

    topic = dds.DynamicData.Topic(participant, "hm-topic", hm_type)
    writer = dds.DynamicData.DataWriter(dds.Publisher(participant), topic)

    sample = dds.DynamicData(hm_type)
    handle = writer.register_instance(sample)

    count = 0
    while count < total_count:
        sample["count"] = count
        sample["name"] = "odd" if count % 2 == 1 else "even"
        print(sample)
        writer.write(sample, handle)
        time.sleep(1)
        count += 1

    writer.unregister_instance(handle)


if __name__ == "__main__":
    dds_domain_id = 0
    total_count = 100
    publisher_main(dds_domain_id, total_count)
