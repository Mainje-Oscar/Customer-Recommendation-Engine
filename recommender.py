"""
Customer Recommendation Engine

"""
from __future__ import annotations

import csv
from pathlib import Path
from collections import Counter, defaultdict

DATA_PATH = Path("data") / "purchases.csv"
OUTPUT_PATH = Path("output") / "recommendations.csv"

def read_purchases(path: Path = DATA_PATH) -> tuple[list[dict], list[dict]]:
    """Return (valid_rows, rejected_rows). Decide what makes a purchase row unusable."""
    """
    Reads purchases from CSV. 
    Rules for valid rows:
    - Must have customer_id, customer_name, and product_id
    - quantity must be an integer > 0
    - Dedupes identical rows to avoid counting the exact same record twice.
    """

    valid_rows = []
    rejected_rows = []
    seen_records = set()

    if not path.exists():
        return valid_rows, rejected_rows

    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {key: (value.strip() if value else "") for key, value in row.items()}
        
            if not row.get("customer_id") or not row.get("product_id") or not row.get("customer_name"):
                rejected_rows.append(row)
                continue
                
            row["customer_id"] = row["customer_id"].upper()

            try:
                qty = int(row["quantity"])
                if qty <= 0:
                    rejected_rows.append(row)
                    continue
            except ValueError:
                rejected_rows.append(row)

            record_tuple = (
                row["customer_id"], row["product_id"], 
                row["purchase_date"], row["quantity"]
            )
            if record_tuple in seen_records:
                rejected_rows.append(row)
                continue
            
            seen_records.add(record_tuple)
            valid_rows.append(row)    

    return valid_rows, rejected_rows


def jaccard(a: set, b: set) -> float:
    """|A intersection B| / |A union B|. Define (and test) what happens when both are empty."""
    """Calculates Jaccard similarity. Returns 0.0 if both sets are empty."""
    if not a or not b:
        return 0.0
    return len(a.intersection(b)) / len(a.union(b))


class RecommendationEngine:
    """Recommends products a customer has not bought, based on similar customers."""
    def __init__(self, purchases: list[dict], min_similarity: float = 0.0) -> None:
        self.min_similarity = min_similarity
        self.customer_products = defaultdict(set)
        self.customer_names = {}
        self.product_data = {}  # product_id -> {"product_name": x, "category": y}
        self.product_popularity = Counter()
        self.category_popularity = Counter()

        for row in purchases:
            c_id = row["customer_id"]
            p_id = row["product_id"]
            
            self.customer_products[c_id].add(p_id)
            self.customer_names[c_id] = row["customer_name"]
            self.product_popularity[p_id] += int(row["quantity"])
            self.category_popularity[row["category"]] += int(row["quantity"])
            
            if p_id not in self.product_data:
                self.product_data[p_id] = {
                    "product_name": row["product_name"],
                    "category": row["category"]
                }
                
        # Sorted popular items fallback list
        self.fallback_products = [
            p_id for p_id, _ in self.product_popularity.most_common() if p_id in self.product_data
        ]

    def history(self, customer_id: str) -> set[str]:
        c_id = customer_id.upper()
        if c_id not in self.customer_products:
            raise KeyError(f"Customer {customer_id} not found.")
        return self.customer_products[c_id]

    def similarity(self, customer_a: str, customer_b: str) -> float:
        return jaccard(self.history(customer_a), self.history(customer_b))

    def similar_customers(self, customer_id: str, k: int = 3) -> list[tuple[str, float]]:
        """Return the k most similar other customers as (customer_id, score), best first."""
        target = customer_id.upper()
        if target not in self.customer_products:
            raise KeyError(f"Customer {customer_id} not found.")

        scores = []
        for other_id in self.customer_products:
            if other_id == target:
                continue
            similar = self.similarity(target, other_id)
            if similar >= self.min_similarity:
                scores.append((other_id, similar))
        
        # Sort by similarity descending, then by customer_id for ties
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:k]

    def candidate_products(self, customer_id: str, k: int = 3) -> dict[str, dict]:
        """Products bought by similar customers but not by this one, with the evidence for each."""
        target = customer_id.upper()
        user_history = self.history(target)
        neighbors = self.similar_customers(target, k=k)
        
        candidates = {}
        for n_id, similar in neighbors:
            if similar == 0.0:
                continue
            for p_id in self.history(n_id):
                if p_id in user_history:
                    continue
                if p_id not in candidates:
                    candidates[p_id] = {"supporting_neighbors": [], "sum_similarity": 0.0}
                candidates[p_id]["supporting_neighbors"].append(n_id)
                candidates[p_id]["sum_similarity"] += similar
                
        return candidates

    def score(self, p_id: str, evidence: dict) -> float:
        """Ranking rule. Combine similarity with at least one other signal."""
        # Rule: Sum of similarity of buying neighbors + 0.001 * item popularity
        return evidence["sum_similarity"] + (0.001 * self.product_popularity.get(p_id, 0))

    def recommend(self, customer_id: str, n: int = 3) -> list[dict]:
        """Return up to n ranked recommendations for one customer."""
        target = customer_id.upper()
        user_history = self.history(target)
        candidates = self.candidate_products(target)
        
        ranked = []
        for p_id, evidence in candidates.items():
            final_score = self.score(p_id, evidence)
            meta = self.product_data.get(p_id, {"product_name": "Unknown", "category": "Unknown"})
            ranked.append({
                "product_id": p_id,
                "product_name": meta["product_name"],
                "category": meta["category"],
                "score": final_score,
                "supporting_customers": sorted(evidence["supporting_neighbors"])
            })
            
        # Sort candidates by score descending, then product_id ascending
        ranked.sort(key=lambda x: (-x["score"], x["product_id"]))
        records = ranked[:n]
        
        # Fallback: If a customer has fewer than n recommendations, fill with store's top popular items they haven't bought
        if len(records) < n:
            for p_id in self.fallback_products:
                if p_id not in user_history and p_id not in [r["product_id"] for r in records]:
                    meta = self.product_data[p_id]
                    records.append({
                        "product_id": p_id,
                        "product_name": meta["product_name"],
                        "category": meta["category"],
                        "score": 0.0,
                        "supporting_customers": []
                    })
                if len(records) == n:
                    break
                    
        return records

    def recommend_all(self, n: int = 3) -> list[dict]:
        all_rows = []
        for c_id in sorted(self.customer_products.keys()):
            records = self.recommend(c_id, n=n)
            name = self.customer_names[c_id]
            if not records:
                all_rows.append({
                    "customer_id": c_id, "customer_name": name, "rank": 1,
                    "product_id": "", "product_name": "", "category": "", "score": 0.0, "supporting_customers": ""
                })
            else:
                for index, r in enumerate(records, start=1):
                    all_rows.append({
                        "customer_id": c_id,
                        "customer_name": name,
                        "rank": index,
                        "product_id": r["product_id"],
                        "product_name": r["product_name"],
                        "category": r["category"],
                        "score": round(r["score"], 3),
                        "supporting_customers": ";".join(r["supporting_customers"])
                    })
        return all_rows

    def export_csv(self, rows: list[dict], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["customer_id", "customer_name", "rank", "product_id", "product_name", "category", "score", "supporting_customers"]
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

