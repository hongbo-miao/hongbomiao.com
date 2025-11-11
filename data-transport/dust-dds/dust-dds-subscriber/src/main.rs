use anyhow::{Context, Result};
use dust_dds::{
    domain::domain_participant_factory::DomainParticipantFactory,
    infrastructure::{
        qos::{DataReaderQos, QosKind},
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

    let subscriber = participant
        .create_subscriber(QosKind::Default, NO_LISTENER, NO_STATUS)
        .context("Failed to create subscriber")?;

    let reader_qos = DataReaderQos {
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

    let reader = subscriber
        .create_datareader::<HmMessage>(
            &topic,
            QosKind::Specific(reader_qos),
            NO_LISTENER,
            NO_STATUS,
        )
        .context("Failed to create data reader")?;

    println!("Waiting for data...");

    loop {
        match reader.read_next_sample() {
            Ok(message_sample) => {
                let data = message_sample
                    .data
                    .context("Failed to extract sample data")?;
                println!("Received: {data:?}");
            }
            Err(_) => {
                thread::sleep(Duration::from_millis(100));
            }
        }
    }
}
