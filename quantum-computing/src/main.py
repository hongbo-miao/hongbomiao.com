import logging

from matplotlib import pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2

logger = logging.getLogger(__name__)


def main() -> None:
    # Create a new circuit with two qubits
    qc = QuantumCircuit(2)

    # Add a Hadamard gate to qubit 0
    qc.h(0)

    # Perform a controlled-X gate on qubit 1, controlled by qubit 0
    qc.cx(0, 1)

    # qc.draw("mpl")

    observables_labels = ["IZ", "IX", "ZI", "XI", "ZZ", "XX"]
    observables = [SparsePauliOp(label) for label in observables_labels]

    backend = FakeAlmadenV2()
    estimator = Estimator(backend)

    # Convert to an ISA circuit and layout-mapped observables.
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)
    mapped_observables = [
        observable.apply_layout(isa_circuit.layout) for observable in observables
    ]

    job = estimator.run([(isa_circuit, mapped_observables)])
    result = job.result()
    logger.info(f"{result = }")

    # Submitted one Pub, so this contains one inner result (and some metadata of its own),
    # which had five observables, so contains information on all five.
    pub_result = job.result()[0]
    logger.info(f"{pub_result = }")

    values = pub_result.data.evs
    logger.info(f"{values = }")

    errors = pub_result.data.stds
    logger.info(f"{errors = }")

    plt.plot(observables_labels, values, "-o")
    plt.xlabel("Observables")
    plt.ylabel("Values")
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
