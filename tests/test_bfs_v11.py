import unittest
from khs import core

class BFSVersionTests(unittest.TestCase):
    def test_qwen_prefers_requested_alternative_in_subfolder(self):
        names=['A/bfs_head_v1_qwen_2.1','B/bfs_head_v1.1_qwen_2.1','Z/bfs_head_v1.1_alternative_qwen_2.1']
        self.assertEqual(core.select_adapter(dict.fromkeys(names,{}),None,'qwen'),names[-1])

    def test_qwen_falls_back_when_new_weights_missing(self):
        names=['A/bfs_head_v1_qwen_2.1','Z/bfs_head_v1.1_qwen_2.1']
        self.assertEqual(core.select_adapter(dict.fromkeys(names,{}),None,'qwen'),names[-1])
        self.assertEqual(core.select_adapter({names[0]:{}},None,'qwen'),names[0])

    def test_alternative_does_not_affect_klein_selection(self):
        klein='folder/bfs_head_v1_flux-klein_9b_step3500_rank128'
        entries={'qwen/bfs_head_v1.1_alternative_qwen_2.1':{},klein:{}}
        self.assertEqual(core.select_adapter(entries,9,'klein'),klein)
