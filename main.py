import numpy as np
import os
import uvicorn

import sys

sys.path.insert(0, "./lib")

from src.nlp_engineer_assignment import (
    count_letters,
    print_line,
    read_inputs,
    score,
    train_classifier,
)

from src.nlp_engineer_assignment.dataset import CharTokenizedDataset


def train_model():
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    ###
    # Setup
    ###

    # Constructs the vocabulary as described in the assignment
    vocabs = [chr(ord("a") + i) for i in range(0, 26)] + [" "]

    ###
    # Train
    ###

    train_inputs = read_inputs(os.path.join(cur_dir, "data", "train.txt"))

    model = train_classifier(
        vocabs=vocabs, train_inputs=train_inputs, num_workers=0, n_epochs=5
    )
    model.save_checkpoint("data/trained_model.ckpt")
    print("Saved Model!")


    ###
    # Test
    ###

    test_inputs = read_inputs(os.path.join(cur_dir, "data", "test.txt"))
    test_dataset = CharTokenizedDataset(sentences=test_inputs, vocab=vocabs)
    model.eval()

    golds = []
    predictions = []
    for encoded_text, gold in test_dataset:
        logits, prediction = model.generate(encoded_text)
        predictions.append(prediction.numpy())
        golds.append(gold.numpy())

    golds = np.stack(golds)
    predictions = np.stack(predictions)

    # Print the first five inputs, golds, and predictions for analysis
    for i in range(5):
        print(f"Input {i+1}: {test_inputs[i]}")
        print(f"Gold {i+1}: {count_letters(test_inputs[i]).tolist()}")
        print(f"Pred {i+1}: {predictions[i].tolist()}")
        print_line()

    print(f"Test Accuracy: {100.0 * score(golds, predictions):.2f}%")
    print_line()


if __name__ == "__main__":
    # train_model()
    uvicorn.run(
        "nlp_engineer_assignment.api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1,
    )
