from datetime import datetime
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionShowTime(Action):
    def name(self) -> Text:
        return "action_get_current_time"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        now = datetime.now()
        dispatcher.utter_message(text=f"{now}")
        return []


class ActionGetShirtSize(Action):
    def name(self) -> Text:
        return "action_get_shirt_size"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        shirt_size = tracker.get_slot("shirt_size")
        if not shirt_size:
            dispatcher.utter_message(text="I don't know your shirt size.")
        else:
            dispatcher.utter_message(text=f"Your shirt size is {shirt_size}!")
        return []
