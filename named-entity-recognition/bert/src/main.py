import logging
import textwrap

from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    model_name = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    nlp = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )

    text = textwrap.dedent("""
    Cristiano Ronaldo dos Santos Aveiro (born 5 February 1985) is a Portuguese professional footballer who plays as a forward for, and captains, both Saudi Pro League club Al-Nassr and the Portugal national team.
    Nicknamed CR7, he is widely regarded as one of the greatest players in history, and has won numerous individual accolades throughout his career, including five Ballon d'Ors, a record three UEFA Men's Player of the Year Awards, four European Golden Shoes, and was named five times the world's best player by FIFA.
    He has won 34 trophies in his career, including five UEFA Champions Leagues and the UEFA European Championship.
    He holds the records for most goals (140) and assists (42) in the Champions League, goals (14) and assists (8) in the European Championship, and most international appearances (225) and international goals (143).
    He is the only player to have scored 100 goals with four different clubs.
    He has made over 1,200 professional career appearances, the most by an outfield player, and has scored over 900 official senior career goals for club and country, making him the top goalscorer of all time.
    """)

    entities = nlp(text)
    for entity in entities:
        logger.info(
            f"{entity['entity_group']}: {entity['word']} (confidence: {entity['score']:.3f})",
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
