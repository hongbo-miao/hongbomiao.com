import time

import rti.connextdds as dds

participant = dds.DomainParticipant(domain_id=0)
topic = dds.StringTopicType.Topic(participant, "hm-topic")
writer = dds.StringTopicType.DataWriter(participant.implicit_publisher, topic)

if __name__ == "__main__":
    i = 0
    while True:
        # create StringTopicType
        writer.write(f"Hello World! #{i}")
        i += 1
        time.sleep(1)
