
import sys
import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np

# Mock the modules before importing the script under test
sys.modules["scratch.load_display_data"] = MagicMock()
sys.modules["scratch.display_outlier_images_spectrum"] = MagicMock()
sys.modules["scratch.display_outlier_images"] = MagicMock()

from scratch.find_similar_anomalies import main, load_records, get_embedding_matrices, find_neighbors

class TestFindSimilarAnomalies(unittest.TestCase):
    
    def test_logic_flow(self):
        # 1. Mock Data
        # 5 Objects: A, B, C, D, E
        object_ids = ["A", "B", "C", "D", "E"]
        embeddings = torch.randn(5, 10)
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Make B very similar to A
        embeddings[1] = embeddings[0] + 0.01 * torch.randn(10)
        embeddings[1] = torch.nn.functional.normalize(embeddings[1], p=2, dim=0)
        
        records = [
            {"object_id": oid, "embedding_hsc_desi": emb}
            for oid, emb in zip(object_ids, embeddings)
        ]
        
        # 2. Test get_embedding_matrices
        matrices, ids = get_embedding_matrices(records)
        self.assertEqual(len(ids), 5)
        self.assertIn("embedding_hsc_desi", matrices)
        matrix = matrices["embedding_hsc_desi"]
        self.assertEqual(matrix.shape, (5, 10))
        
        # 3. Test find_neighbors for A (should find B as top neighbor)
        # n_similar = 1
        neighbors = find_neighbors(["A"], object_ids, matrix, n_similar=1)
        # Expected: [A, B] (or whatever is closest, but B is engineered to be close)
        self.assertEqual(len(neighbors), 2)
        self.assertEqual(neighbors[0], "A")
        # Probability of B being closest is high, but let's check structure more than value
        
        # 4. Test Main Execution Flow with Mocks
        with patch("scratch.find_similar_anomalies.load_records") as mock_load:
            with patch("scratch.find_similar_anomalies.EuclidDESIDataset") as mock_ds:
                with patch("scratch.find_similar_anomalies.collect_samples") as mock_collect:
                    with patch("scratch.find_similar_anomalies.plot_vertical_panels") as mock_plot:
                        
                        mock_load.return_value = records
                        
                        # Mock collect_samples to return one dict per requested ID
                        def side_effect_collect(ds, ids, verbose=False):
                            return [{"object_id": i, "image": np.zeros((10,10))} for i in ids]
                        mock_collect.side_effect = side_effect_collect
                        
                        # run main
                        # input file argument is dummy
                        # object_ids argument provided
                        argv = [
                            "--input", "dummy.pt",
                            "--object_ids", "A",
                            "--n-similar", "2"
                        ]
                        
                        main(argv)
                        
                        # Verify steps
                        mock_load.assert_called_with(unittest.mock.ANY)
                        # We asked for A, n=2. Should find A + 2 neighbors = 3 objects.
                        # plot should be called with list of size 3
                        # And it should be called ONCE because we only have 1 embedding type in mock records
                        self.assertEqual(mock_plot.call_count, 1)
                        call_args = mock_plot.call_args
                        samples_passed = call_args[0][0]
                        cols_passed = call_args[1].get('cols') or call_args[0][1]
                        
                        self.assertEqual(len(samples_passed), 3)
                        self.assertEqual(cols_passed, 3) # n_similar + 1
                        
                        # Verify annotation
                        self.assertTrue("[QUERY]" in str(samples_passed[0]["object_id"]))
                        self.assertTrue("[NEIGHBOR 1]" in str(samples_passed[1]["object_id"]))
                        self.assertTrue("[NEIGHBOR 2]" in str(samples_passed[2]["object_id"]))

if __name__ == "__main__":
    unittest.main()
