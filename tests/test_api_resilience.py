import unittest

from cli.main import build_analysis_attempt_configs, is_transient_api_error


class FakeApiError(Exception):
    def __init__(self, message="", status_code=None, code=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class ApiResilienceTests(unittest.TestCase):
    def test_transient_error_detects_retryable_status_codes(self):
        self.assertTrue(is_transient_api_error(FakeApiError("busy", status_code=503)))
        self.assertTrue(is_transient_api_error(FakeApiError("too many requests", code=429)))

    def test_transient_error_detects_retryable_message_patterns(self):
        err = RuntimeError("ServerError: 503 UNAVAILABLE. This model is currently experiencing high demand.")
        self.assertTrue(is_transient_api_error(err))

    def test_non_transient_error_is_not_misclassified(self):
        err = FakeApiError("invalid api key provided", status_code=401)
        self.assertFalse(is_transient_api_error(err))

    def test_google_attempt_configs_include_stable_fallbacks(self):
        config = {
            "llm_provider": "google",
            "deep_think_llm": "gemini-3.1-pro-preview",
            "quick_think_llm": "gemini-3.1-flash-lite-preview",
            "google_thinking_level": "high",
        }

        attempts = build_analysis_attempt_configs(config)
        summaries = [
            (
                item["config"]["deep_think_llm"],
                item["config"]["quick_think_llm"],
                item["config"].get("google_thinking_level"),
                item["wait_seconds"],
                item["reason"],
            )
            for item in attempts
        ]

        self.assertEqual(
            summaries[0],
            ("gemini-3.1-pro-preview", "gemini-3.1-flash-lite-preview", "high", 0, "primary"),
        )
        self.assertIn(
            ("gemini-3.1-pro-preview", "gemini-3.1-flash-lite-preview", "high", 15, "primary"),
            summaries,
        )
        self.assertIn(
            ("gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite-preview", None, 15, "fallback"),
            summaries,
        )
        self.assertIn(
            ("gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite-preview", None, 60, "fallback"),
            summaries,
        )

    def test_non_google_provider_keeps_single_attempt(self):
        config = {
            "llm_provider": "openai",
            "deep_think_llm": "gpt-5.4",
            "quick_think_llm": "gpt-5.4-mini",
        }

        self.assertEqual(
            build_analysis_attempt_configs(config),
            [{"config": config, "wait_seconds": 0, "reason": "primary"}],
        )


if __name__ == "__main__":
    unittest.main()
