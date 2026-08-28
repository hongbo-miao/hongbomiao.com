from config import config
from dbos import DBOS, DBOSConfig, SetWorkflowID

# DBOS checkpoints workflow and step state into this Postgres database, so a
# crashed process can resume a workflow from its last completed step instead
# of starting over.
dbos_config: DBOSConfig = {
    "name": "hm-dbos",
    "application_version": "1.0.0",
    "system_database_url": config.DBOS_SYSTEM_DATABASE_URL,
}


@DBOS.step(retries_allowed=True, max_attempts=3, interval_seconds=1.0, backoff_rate=2.0)
def fetch_exchange_rate(currency_code: str) -> float:
    # Fails on the first two attempts so the retry backoff is visible in the logs.
    attempt_number = DBOS.step_status.current_attempt if DBOS.step_status else None
    if attempt_number is not None and attempt_number < 2:
        error_message = f"Simulated transient failure fetching rate for {currency_code}"
        raise RuntimeError(error_message)
    exchange_rate = 1.08
    DBOS.logger.info(f"Fetched exchange rate for {currency_code}: {exchange_rate}")
    return exchange_rate


@DBOS.step()
def record_payment(amount: float, exchange_rate: float) -> float:
    converted_amount = round(amount * exchange_rate, 2)
    DBOS.logger.info(f"Recorded payment of {amount} converted to {converted_amount}")
    return converted_amount


@DBOS.workflow()
def process_payment_workflow(amount: float, currency_code: str) -> float:
    # Workflow bodies must be deterministic: no I/O, randomness, or clock
    # reads here. All of that belongs in steps, which are checkpointed
    # independently and re-run at most once each after a crash.
    exchange_rate = fetch_exchange_rate(currency_code)
    DBOS.sleep(2)
    return record_payment(amount, exchange_rate)


if __name__ == "__main__":
    DBOS(config=dbos_config)
    DBOS.launch()

    # A fixed workflow ID is an idempotency key: running this script again
    # with the same ID returns the recorded result instead of re-executing
    # the steps.
    with SetWorkflowID("process-payment-example"):
        result = process_payment_workflow(100.0, "EUR")

    DBOS.logger.info(f"Workflow result: {result}")
    DBOS.destroy()
