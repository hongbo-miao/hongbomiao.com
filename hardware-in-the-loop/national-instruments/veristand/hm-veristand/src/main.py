from niveristand import NivsParam, nivs_rt_sequence, realtimesequencetools
from niveristand.clientapi import BooleanValue, ChannelReference, DoubleValue
from niveristand.library import wait


@nivs_rt_sequence
@NivsParam("engine_power", BooleanValue(0), NivsParam.BY_REF)
@NivsParam("desired_rpm", DoubleValue(0), NivsParam.BY_REF)
def engine_demo_basic(engine_power, desired_rpm) -> None:  # noqa: ANN001
    engine_power_chan = ChannelReference("Aliases/EnginePower")
    desired_rpm_chan = ChannelReference("Aliases/DesiredRPM")
    engine_power_chan.value = engine_power.value
    desired_rpm_chan.value = desired_rpm.value
    wait(DoubleValue(20))
    engine_power_chan.value = False
    desired_rpm_chan.value = 0


@nivs_rt_sequence
def run_engine_demo() -> None:
    engine_demo_basic(BooleanValue(True), DoubleValue(2500))  # noqa: FBT003


def run_non_deterministic() -> None:
    print("Started non-deterministic")
    run_engine_demo()
    print("Finished non-deterministic")


def run_deterministic() -> None:
    print("Started deterministic")
    realtimesequencetools.run_py_as_rtseq(run_engine_demo)
    print("Finished deterministic")


def main(real_time_sequence_path: str) -> None:
    realtimesequencetools.save_py_as_rtseq(run_engine_demo, real_time_sequence_path)
    run_non_deterministic()
    run_deterministic()


if __name__ == "__main__":
    external_real_time_sequence_path = (
        "c:\\hongbomiao.com\\hardware-in-the-loop\\national-instruments\\hm-veristand"
    )
    main(external_real_time_sequence_path)
