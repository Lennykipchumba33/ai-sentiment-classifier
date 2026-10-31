import unittest

from src.sentiment_classifier import predict


class SentimentClassifierTests(unittest.TestCase):
    def test_positive_text(self):
        label, confidence = predict(["The helpful team made me very happy"])[0]
        self.assertEqual(label, "positive")
        self.assertGreater(confidence, 0.5)

    def test_negative_text(self):
        label, confidence = predict(["The buggy app was frustrating and terrible"])[0]
        self.assertEqual(label, "negative")
        self.assertGreater(confidence, 0.5)

    def test_five_real_world_examples(self):
        examples = [
            ("The grocery delivery arrived early and every item was fresh.", "positive"),
            ("The bank app made it easy to freeze my card in seconds.", "positive"),
            ("My train was cancelled with no warning and support never replied.", "negative"),
            ("The headphones stopped charging after two days and the replacement process was frustrating.", "negative"),
            ("The hotel room was clean, quiet, and exactly as described.", "positive"),
        ]

        predictions = predict([text for text, _ in examples])
        for (text, expected), (actual, confidence) in zip(examples, predictions):
            with self.subTest(text=text):
                self.assertEqual(actual, expected)
                self.assertGreater(confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
