from datetime import datetime
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionShowTime(Action):
    def name(self) -> Text:
        return "action_show_time"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        now = datetime.now()
        time = now.strftime("%H:%M:%S")
        dispatcher.utter_message(text=f"{time}")
        return []
