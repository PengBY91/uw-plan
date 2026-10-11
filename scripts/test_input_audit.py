import unittest
from audit_input_trace import stream_summary

class InputAuditTests(unittest.TestCase):
    def test_complete(self):
        result = stream_summary([100,105,110],[100,105,110])
        self.assertTrue(result['ordered_complete'])
        self.assertEqual(result['missing_count'],0)
    def test_missing_and_duplicate_cannot_cancel(self):
        result = stream_summary([100,105,110],[100,100,110])
        self.assertEqual(result['missing_count'],1)
        self.assertEqual(result['duplicate_count'],1)
        self.assertFalse(result['ordered_complete'])
    def test_reordered_complete_set_is_not_complete_stream(self):
        result = stream_summary([100,105,110],[100,110,105])
        self.assertEqual(result['missing_count'],0)
        self.assertEqual(result['non_increasing_count'],1)
        self.assertFalse(result['ordered_complete'])
    def test_empty_received(self):
        self.assertEqual(stream_summary([100],[])['received_fraction'],0)

if __name__ == '__main__': unittest.main()
