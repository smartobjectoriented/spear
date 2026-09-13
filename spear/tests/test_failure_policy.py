import unittest

from failure_policy import (
    Failure, FailureKind, RetryAction, RetryPolicy, classify_tool_failure,
)


class FailurePolicyTests(unittest.TestCase):
    def test_explicit_retryable_model_failure_is_bounded(self):
        policy = RetryPolicy({FailureKind.MODEL_ERROR: 1})
        failure = Failure(FailureKind.MODEL_ERROR, "temporary")
        self.assertTrue(policy.decide(failure, 0).should_retry)
        self.assertFalse(policy.decide(failure, 1).should_retry)

    def test_non_retryable_override(self):
        policy = RetryPolicy({FailureKind.MODEL_ERROR: 4})
        decision = policy.decide(Failure(FailureKind.MODEL_ERROR, "bad",
                                         retryable=False), 0)
        self.assertEqual(decision.action, RetryAction.DO_NOT_RETRY)

    def test_permission_is_not_blindly_retried(self):
        policy = RetryPolicy()
        failure = Failure(FailureKind.PERMISSION_DENIED, "denied")
        self.assertFalse(policy.decide(failure, 0).should_retry)

    def test_invalid_turn_uses_bounded_reprompt(self):
        decision = RetryPolicy().decide(
            Failure(FailureKind.INVALID_MODEL_TURN, "malformed"), 0)
        self.assertEqual(decision.action, RetryAction.REPROMPT)

    def test_grounded_tool_classification(self):
        self.assertEqual(classify_tool_failure("denied", None, None),
                         FailureKind.PERMISSION_DENIED)
        self.assertEqual(classify_tool_failure("failed", None, 2),
                         FailureKind.COMMAND_FAILED)
        self.assertEqual(classify_tool_failure("timeout", None, None),
                         FailureKind.TIMEOUT)

