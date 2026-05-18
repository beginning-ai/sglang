import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager


class TestQwen35TextOnly(unittest.TestCase):
    def test_text_only_multimodal_payload_crashes_fast(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.model_config = SimpleNamespace(is_qwen3_5_text_only=True)
        manager._subprocess_watchdog = Mock()
        manager.dump_requests_before_crash = Mock()

        request = GenerateReqInput(text="hello", image_data="fake-image")

        with patch(
            "sglang.srt.managers.tokenizer_manager.kill_process_tree"
        ) as kill_process_tree:
            with self.assertRaises(RuntimeError):
                manager._crash_on_qwen3_5_text_only_multimodal_payload(request)

        manager._subprocess_watchdog.stop.assert_called_once_with()
        manager.dump_requests_before_crash.assert_called_once_with()
        kill_process_tree.assert_called_once_with(ANY, include_parent=True)


if __name__ == "__main__":
    unittest.main()
