import unittest
import process_data
import os

class TestProcessData(unittest.TestCase):
    def test_grab_and_split_data(self):
        print(process_data.grab_and_split_data("", ""))
        # self.assertRaises(FileNotFoundError, process_data.grab_and_split_data("", "", None))


if __name__ == "__main__":
    unittest.main()