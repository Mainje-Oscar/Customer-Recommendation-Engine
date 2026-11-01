"""
Unit tests for the Recommendation Engine
"""
import unittest
from pathlib import Path
from recommender import jaccard, read_purchases, RecommendationEngine

class TestRecommendationEngine(unittest.TestCase):

    def setUp(self):
        """Dummy transaction data for testing."""
        self.mock_valid_purchases = [
            {"customer_id": "C101", "customer_name": "Mainje", "product_id": "P01", 
             "product_name": "Milk", "category": "Dairy", "purchase_date": "2026-10-01", "quantity": "2"},
            {"customer_id": "C101", "customer_name": "Mainje", "product_id": "P02", 
             "product_name": "Bread", "category": "Bakery", "purchase_date": "2026-10-01", "quantity": "1"},
            {"customer_id": "C102", "customer_name": "Oscar", "product_id": "P01", 
             "product_name": "Milk", "category": "Dairy", "purchase_date": "2026-10-01", "quantity": "1"},
            {"customer_id": "C102", "customer_name": "Oscar", "product_id": "P03", 
             "product_name": "Eggs", "category": "Fresh", "purchase_date": "2026-10-02", "quantity": "3"},
            {"customer_id": "C103", "customer_name": "Nerina", "product_id": "P02", 
             "product_name": "Bread", "category": "Bakery", "purchase_date": "2026-10-03", "quantity": "1"},
        ]
        self.engine = RecommendationEngine(self.mock_valid_purchases)

    def test_jaccard_similarity_calculation(self):
        """Verify Jaccard outputs and boundaries."""
        set_a = {"P01", "P02"}
        set_b = {"P01", "P03"}
        self.assertAlmostEqual(jaccard(set_a, set_b), 1 / 3)

        # Matching profiles
        self.assertEqual(jaccard(set_a, set_a), 1.0)

        # No items in common
        self.assertEqual(jaccard({"P01"}, {"P02"}), 0.0)

        self.assertEqual(jaccard(set(), set()), 0.0)

    def test_id_case_normalization(self):
        """Ensure lookups are successful regardless of lowercase entries."""
        self.assertEqual(self.engine.history("c103"), {"P02"})
        self.assertEqual(self.engine.history("C103"), {"P02"})

    def test_unknown_customer_exception_handling(self):
        """Confirm that unauthorized profile lookups causes KeyErrors."""
        with self.assertRaises(KeyError):
            self.engine.history("C999")
            
        with self.assertRaises(KeyError):
            self.engine.similar_customers("C999")

    def test_scoring_and_ranking_rule_execution(self):
        records = self.engine.recommend("C101", n=1)
        
        self.assertEqual(records[0]["product_id"], "P03")
        self.assertEqual(records[0]["supporting_customers"], ["C102"])
        
        # Jaccard(Mainje, Oscar) = 1/3 (Intersection: P01, Union: P01, P02, P03)
        # P03 quantity popularity score in setUp is 3 checkouts
        # Expected Score = (1/3) + (0.001 * 3) = 0.3333333 + 0.003 = 0.3363333
        self.assertAlmostEqual(records[0]["score"], (1/3) + 0.003)

    def test_fallback_strategy_for_unsupported_customers(self):
        recs = self.engine.recommend("C103", n=2)
        
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0]["product_id"], "P01") 
        self.assertEqual(recs[1]["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
