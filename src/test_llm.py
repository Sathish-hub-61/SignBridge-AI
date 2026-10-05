from llm import interpret_and_reply

tests = [
    [("help", 0.95), ("pain", 0.92), ("doctor", 0.90)],   # confident, urgent
    [("water", 0.88), ("where", 0.91)],                    # simple question
    [("medicine", 0.60), ("pain", 0.93)],                  # low confidence -> must ask
    [("yes", 0.97)],                                       # needs context
]
for words in tests:
    print(words)
    print(interpret_and_reply(words, []), "\n")