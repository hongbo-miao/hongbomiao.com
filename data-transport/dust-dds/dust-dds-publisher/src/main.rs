use anyhow::{Context, Result};
use dust_dds::{
    domain::domain_participant_factory::DomainParticipantFactory,
    infrastructure::{
        qos::{DataWriterQos, QosKind},
        qos_policy::{
            DurabilityQosPolicy, DurabilityQosPolicyKind, HistoryQosPolicy, HistoryQosPolicyKind,
            ReliabilityQosPolicy, ReliabilityQosPolicyKind,
        },
        status::NO_STATUS,
        time::DurationKind,
    },
    listener::NO_LISTENER,
};
use std::{thread, time::Duration};

include!(concat!(env!("OUT_DIR"), "/hm_message.rs"));

fn main() -> Result<()> {
    let domain_id = 0;
    let participant_factory = DomainParticipantFactory::get_instance();

    let participant = participant_factory
        .create_participant(domain_id, QosKind::Default, NO_LISTENER, NO_STATUS)
        .context("Failed to create domain participant")?;

    let topic = participant
        .create_topic::<HmMessage>(
            "HmMessage",
            "HmMessage",
            QosKind::Default,
            NO_LISTENER,
            NO_STATUS,
        )
        .context("Failed to create topic")?;

    let publisher = participant
        .create_publisher(QosKind::Default, NO_LISTENER, NO_STATUS)
        .context("Failed to create publisher")?;

    let writer_qos = DataWriterQos {
        reliability: ReliabilityQosPolicy {
            kind: ReliabilityQosPolicyKind::Reliable,
            max_blocking_time: DurationKind::Finite(dust_dds::infrastructure::time::Duration::new(
                1, 0,
            )),
        },
        durability: DurabilityQosPolicy {
            kind: DurabilityQosPolicyKind::TransientLocal,
        },
        history: HistoryQosPolicy {
            kind: HistoryQosPolicyKind::KeepLast(10),
        },
        ..Default::default()
    };

    let writer = publisher
        .create_datawriter::<HmMessage>(
            &topic,
            QosKind::Specific(writer_qos),
            NO_LISTENER,
            NO_STATUS,
        )
        .context("Failed to create data writer")?;

    println!("Publishing messages...");

    let mut count = 0;
    loop {
        count += 1;

        let message = HmMessage {
            message: "Hello, World!".to_string(),
            count,
        };

        writer
            .write(message, None)
            .context("Failed to write message")?;

        println!("Published message: message=Hello, World!, count={count}");

        thread::sleep(Duration::from_secs(1));
    }
}
