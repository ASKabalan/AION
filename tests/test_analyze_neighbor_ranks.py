
import sys
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

# Mock modules
sys.modules["scratch.find_similar_anomalies"] = MagicMock()
sys.modules["scratch.display_outlier_images"] = MagicMock()

# Import after mocking
from scratch.analyze_neighbor_ranks import main

class TestAnalyzeNeighborRanks(unittest.TestCase):
    
    def test_flow(self):
        # Mock dependencies
        with patch("scratch.analyze_neighbor_ranks.load_records") as mock_load_recs:
            with patch("scratch.analyze_neighbor_ranks.load_scores") as mock_load_scores:
                with patch("scratch.analyze_neighbor_ranks.get_embedding_matrices") as mock_get_mats:
                    with patch("scratch.analyze_neighbor_ranks.find_neighbors") as mock_find:
                        with patch("scratch.analyze_neighbor_ranks.plot_rank_distribution") as mock_plot:
                            with patch("scratch.analyze_neighbor_ranks.read_object_ids") as mock_read_ids:
                                
                                # Setup return values
                                mock_load_recs.return_value = [{"obj": 1}]
                                
                                # Scores DF: 5 objects with ranks
                                scores_df = pd.DataFrame({
                                    "object_id": ["A", "B", "C", "D", "E"],
                                    "embedding_key": ["embedding_hsc_desi"] * 5,
                                    "rank": [10, 20, 30, 40, 50]
                                })
                                mock_load_scores.return_value = scores_df
                                
                                # Matrices
                                mock_get_mats.return_value = ({"embedding_hsc_desi": np.zeros((5,5))}, ["A","B","C","D","E"])
                                
                                # Find neighbors: Query A -> Neighbor B, C
                                # Output: [A, B, C]
                                mock_find.return_value = ["A", "B", "C"]
                                
                                # Input args
                                argv = [
                                    "--input", "dummy.pt",
                                    "--scores", "dummy.csv",
                                    "--object_ids", "A",
                                    "--n-similar", "2"
                                ]
                                
                                main(argv)
                                
                                # Verification
                                mock_load_recs.assert_called()
                                mock_load_scores.assert_called()
                                mock_find.assert_called()
                                
                                # Check plot called
                                # Should receive ranks for B (20) and C (30)
                                self.assertTrue(mock_plot.called)
                                args, _ = mock_plot.call_args
                                ranks_passed = args[0]
                                self.assertEqual(len(ranks_passed), 2)
                                self.assertIn(20, ranks_passed)
                                self.assertIn(30, ranks_passed)

if __name__ == "__main__":
    unittest.main()
