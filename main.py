"""
Engine execution script
"""
from pathlib import Path
from recommender import read_purchases, RecommendationEngine

def main():
    data_file = Path("data") / "purchases.csv"
    output_file = Path("output") / "recommendations.csv"

    valid_rows, rejected_rows = read_purchases(data_file)

    engine = RecommendationEngine(valid_rows)
    
    target_check = "C113"
    if target_check in engine.customer_products:
        history_names = {engine.product_data[p]["product_name"] for p in engine.history(target_check)}
        print()
        print(f"Customer {target_check} ({engine.customer_names[target_check]}) bought: {history_names}")
        print("Most similar customers:")
        for sim_id, score in engine.similar_customers(target_check, k=3):
            print(f"  {sim_id}  jaccard={score:.2f}")
            
        print("\nRecommendations:")
        for index, record in enumerate(engine.recommend(target_check, n=3), start=1):
            print(
                f"  {index}. {record['product_name']} ({record['category']}) "
                f" score={record['score']:.3f} because "
                f"{len(record['supporting_customers'])} similar customers bought it"
            )
        print()

    all_recs = engine.recommend_all(n=3)
    engine.export_csv(all_recs, output_file)

    print("Run Summary")
    print(f"Purchases read: {len(valid_rows) + len(rejected_rows)} | used: {len(valid_rows)} | rejected: {len(rejected_rows)}")
    print(f"Customers: {len(engine.customer_products)} | Products: {len(engine.product_data)} | Recommendations written: {len(all_recs)}")

if __name__ == "__main__":
    main()
