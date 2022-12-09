from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("I've been waiting for this course my whole life."))
print(classifier("I've been waiting for so long for this package to ship."))
