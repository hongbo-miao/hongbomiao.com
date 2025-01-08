from datetime import datetime
from typing import Any

from rasa_sdk import Action, FormValidationAction, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict


class ActionShowTime(Action):
    def name(self) -> str:
        return "action_get_current_time"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: dict[str, Any],
    ) -> list[dict[str, Any]]:
        now = datetime.now()
        dispatcher.utter_message(text=f"{now}")
        return []


class ActionGetShirtSize(Action):
    def name(self) -> str:
        return "action_get_my_favorite_color"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: dict[str, Any],
    ) -> list[dict[str, Any]]:
        my_favorite_color = tracker.get_slot("my_favorite_color")
        if not my_favorite_color:
            dispatcher.utter_message(text="I don't know your favorite color.")
        else:
            dispatcher.utter_message(
                text=f"Your favorite color is {my_favorite_color}!",
            )
        return []


ALLOWED_PIZZA_SIZES = ["small", "medium", "large"]
ALLOWED_PIZZA_TYPES = ["cheese", "hawaiian", "pepperoni"]


class ValidateSimplePizzaForm(FormValidationAction):
    def name(self) -> str:
        return "validate_pizza_form"

    def validate_pizza_size(
        self,
        slot_value: str,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> dict[str, str | None]:
        if slot_value.lower() not in ALLOWED_PIZZA_SIZES:
            dispatcher.utter_message(
                text=f"We only accept pizza sizes: {'/'.join(ALLOWED_PIZZA_SIZES)}.",
            )
            return {"pizza_size": None}
        dispatcher.utter_message(text=f"OK! You want to have a {slot_value} pizza.")
        return {"pizza_size": slot_value}

    def validate_pizza_type(
        self,
        slot_value: str,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> dict[str, str | None]:
        if slot_value not in ALLOWED_PIZZA_TYPES:
            dispatcher.utter_message(
                text=f"We only serve {'/'.join(ALLOWED_PIZZA_TYPES)}.",
            )
            return {"pizza_type": None}
        dispatcher.utter_message(text=f"OK! You want to have a {slot_value} pizza.")
        return {"pizza_type": slot_value}
