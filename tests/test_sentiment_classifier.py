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


if __name__ == "__main__":
    unittest.main()
