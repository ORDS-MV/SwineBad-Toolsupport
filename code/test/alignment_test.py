import unittest
import sys

sys.path.insert(1, '../util')
from alignment import align_blocks

class TestAlignBlocks(unittest.TestCase):
    def test_basic_alignment_high_threshold(self):
        block1 = ["hello world", "this is a test", "another line", "final line"]
        block2 = ["hello world", "this is", "another", "final line"]
        aligned_block1, aligned_block2 = align_blocks(block1, block2, threshold=7)

        # Expected alignment: second lines partially match, third lines partially match
        expected_block1 = ["hello world", "this is a test", "another line", "final line"]
        expected_block2 = ["hello world","this is", "another", "final line"]

        self.assertEqual(aligned_block1, expected_block1)
        self.assertEqual(aligned_block2, expected_block2)

    def test_basic_alignment_medium_threshold(self):
        block1 = ["hello world", "this is a test", "another line", "final line"]
        block2 = ["hello world", "this is", "another", "final line"]
        aligned_block1, aligned_block2 = align_blocks(block1, block2, threshold=5)

        # Expected alignment: second lines partially match, third lines partially match
        expected_block1 = ["hello world", "this is a test", "another line", "final line",""]
        expected_block2 = ["hello world","","another", "final line","this is"]

        self.assertEqual(aligned_block1, expected_block1)
        self.assertEqual(aligned_block2, expected_block2)

    def test_basic_alignment_small_threshold(self):
        block1 = ["hello world", "this is a test", "another line", "final line"]
        block2 = ["hello world", "this is", "another", "final line"]
        aligned_block1, aligned_block2 = align_blocks(block1, block2, threshold=1)

        # Expected alignment: second lines partially match, third lines partially match: for both -> threshold is too low
        expected_block1 = ["hello world", "this is a test", "another line", "final line","",""]
        expected_block2 = ["hello world","","", "final line","this is","another"]

        self.assertEqual(aligned_block1, expected_block1)
        self.assertEqual(aligned_block2, expected_block2)

    def test_mismatched_length(self):
        block1 = ["line 1", "line 2", "line 3"]
        block2 = ["line 1", "line 2", "line 3", "extra line"]
        aligned_block1, aligned_block2 = align_blocks(block1, block2,threshold=0)

        # Expected: last line in block2 is unmatched
        expected_block1 = ["line 1", "line 2", "line 3", ""]
        expected_block2 = ["line 1", "line 2", "line 3", "extra line"]

        self.assertEqual(aligned_block1, expected_block1)
        self.assertEqual(aligned_block2, expected_block2)

    def test_mismatched_length_reverse(self):
        block1 = ["line 1", "line 2", "line 3", "extra line"]
        block2 = ["line 1", "line 2", "line 3"]
        aligned_block1, aligned_block2 = align_blocks(block1, block2,threshold=0)

        # Expected: last line in block1 is unmatched
        expected_block1 = ["line 1", "line 2", "line 3", "extra line"]
        expected_block2 = ["line 1", "line 2", "line 3", ""]

        self.assertEqual(aligned_block1, expected_block1)
        self.assertEqual(aligned_block2, expected_block2)

    def test_insert_blanks_for_missing_lines(self):
        block1 = ["line 1", "line 2", "line 4"]
        block2 = ["line 1", "line 3", "line 4"]
        aligned_block1, aligned_block2 = align_blocks(block1, block2,threshold=0)

        # Expected: "line 2" in block1 has no match in block2; "line 3" in block2 has no match in block1
        expected_block1 = ["line 1", "line 2", "line 4", ""]
        expected_block2 = ["line 1", "", "line 4", "line 3"]

        self.assertEqual(aligned_block1, expected_block1)
        self.assertEqual(aligned_block2, expected_block2)


    def test_insert_blanks_for_missing_lines_reverse(self):
        block1 = ["line 1", "line 3", "line 4"]
        block2 = ["line 1", "line 2", "line 4"]
        aligned_block1, aligned_block2 = align_blocks(block1, block2,threshold=0)

        # Expected: "line 2" in block2 has no match in block1; "line 3" in block1 has no match in block2
        expected_block1 = ["line 1", "line 3", "line 4", ""]
        expected_block2 = ["line 1", "", "line 4", "line 2"]

        self.assertEqual(aligned_block1, expected_block1)
        self.assertEqual(aligned_block2, expected_block2)

    def test_insert_blanks_for_missing_lines_single_sided(self):
        block1 = ["line 1", "line 2","line 3", "line 4"]
        block2 = ["line 1", "line 3", "line 4"]
        aligned_block1, aligned_block2 = align_blocks(block1, block2, threshold=0)

        # Expected: block1 is complete; "line 2" in block1 has no match in block2
        expected_block1 = ["line 1","line 2","line 3","line 4"]
        expected_block2 = ["line 1", "", "line 3", "line 4"]

        self.assertEqual(aligned_block1, expected_block1)
        self.assertEqual(aligned_block2, expected_block2)

    def test_error(self):
        block1 = ["line 1", "line 2","line 3", "line 4"]
        block2 = ["line 1","error","line 3", "line 4"]
        aligned_block1, aligned_block2 = align_blocks(block1, block2, threshold=0)

        # Expected: block1 is complete; "line 2" in block1 has no match in block2
        expected_block1 = ["line 1","line 2","line 3","line 4",""]
        expected_block2 = ["line 1", "", "line 3", "line 4","error"]

        self.assertEqual(aligned_block1, expected_block1)
        self.assertEqual(aligned_block2, expected_block2)

    def test_all_blank_fillings(self):
        block1 = ["line 1", "line 2"]
        block2 = ["line A", "line B"]
        aligned_block1, aligned_block2 = align_blocks(block1, block2,threshold=0)

        # Expected: Since none of the lines match, they should each get blanks
        expected_block1 = ["line 1", "line 2", "",""]
        expected_block2 = ["","", "line A", "line B"]

        self.assertEqual(aligned_block1, expected_block1)
        self.assertEqual(aligned_block2, expected_block2)

    def test_identical_blocks(self):
        block1 = ["same line 1", "same line 2", "same line 3"]
        block2 = ["same line 1", "same line 2", "same line 3"]
        aligned_block1, aligned_block2 = align_blocks(block1, block2,threshold=0)

        # Expected: blocks are identical, so no blanks are needed
        self.assertEqual(aligned_block1, block1)
        self.assertEqual(aligned_block2, block2)


if __name__ == '__main__':
    unittest.main()