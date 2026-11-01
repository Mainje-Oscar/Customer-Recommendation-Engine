# Customer Recommendation Engine

A Python recommendation engine built for neighborhood grocery store checkouts across Nairobi. This engine implements Jaccard similarity metrics alongside custom popularity filters to give product suggestions without requiring machine learning platforms.

## Project Structure
```text
day4-recommendation-engine/
├── recommender.py           # Core logic (Jaccard similarity and Engine Class)
├── main.py                  # Script execution entry point and console logger
├── test_recommender.py      # Unit testing logic
├── data/
│   └── purchases.csv        # Raw transaction history
├── output/
│   └── recommendations.csv  # Generated output file
└── README.md                
```

## Setup and execution

### Run the main script
To clean the data, process current transactions, display findings, and export recommendations, run:
```bash
python main.py
```

### Run verification tests
To run the logic testing data cleaning filters, lookups, and edge-case exceptions, run:
```bash
python -m unittest test_recommender.py
```

---

## Data cleaning strategy
* **ID Case correction:** Normalizes tracking markers (e.g., `c106` and `C106`) into uppercase strings to prevent fragmented profiles.
* **Data integrity filters:** Drops raw rows missing essential tags (`customer_id`, `customer_name`, or `product_id`).
* **Quantity checks:** Discards checkouts with invalid values or zero counts (`quantity <= 0`) since they do not reflect actual product purchases.
* **Deduplication:** Filters out exact identical records occurring on the same transaction date to eliminate wrong scoring bias.

---

## Scoring logic (Worked Example)
To keep suggestions easily explainable to shop owners, the engine applies a multi-signal scoring rule combining peer interest with performance metrics of the entire store:

$$\text{Score} = \sum (\text{Jaccard similarity of supporting neighbors}) + (0.001 \times \text{Product purchase count of the entire store})$$

### Worked Example (Customer C113 - Wambui)
* **Wambui's purchase history:** `P01` (Eggs), `P02` (Milk), `P03` (Bread), `P04` (Blue Band), `P05` (Kimbo).
* **Identified neighbors:** `C102` (Jaccard similarity = 0.40) and `C114` (Jaccard similarity = 0.40).
* **Candidate evaluation:** Both neighbors previously bought `P06` (Pishori Rice 5kg), which Wambui has never purchased. 
* **Store-wide popularity:** In the store, across the dataset, `P06` has a accumulated quantity score of 6 checkouts.
* **Final score calculation:**
$$\text{Score} = (0.40 + 0.40) + (0.001 \times 6) = 0.806$$

---

## Handling edge cases

### 1. New customer profiles (Fallback)
If a customer has a completely new history, an empty basket, or zero shared interests with the rest of the neighborhood (resulting in a Jaccard score of `0.0`), the system executes a **popular item fallback**. It backfills the remaining empty recommendation slots with the store's top trending items that the customer does not currently own, marking them with a score of `0.000`.

### 2. Tie-breaking
When multiple products share identical neighborhood support weights, ties are broken by sorting the candidates in ascending order by their unique `product_id`. This prevents random positional shifting during re-runs.

### 3. Unknown ID queries
Passing an unrecognized identifier into lookup methods triggers a standard `KeyError` with descriptive messaging instead of failing silently or returning confusing fallback arrays.
