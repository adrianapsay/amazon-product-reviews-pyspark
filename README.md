# Large-Scale Feature Engineering & Predictive Modeling with PySpark  

## 📌 Overview  
This is a scalable **end-to-end data processing and machine learning pipeline** using **PySpark DataFrames** on a distributed Spark cluster (UCSD DSMLP). Working with **12GB of Amazon product and review data**, the pipeline performs complex feature engineering, builds ML-ready representations, and trains/tunes a Decision Tree Regressor for product rating prediction.  

Doing this helped me understand more in-depth **big data processing, feature engineering, distributed ML training, and runtime optimization**. It was originally completed as part of UCSD’s *Systems for Scalable Analytics* coursework. 

---

## ⚙️ Features  

### Data Processing  
- Flattened nested JSON structures (categories, salesRank, also_viewed, related)  
- Performed **cross-table joins** between reviews and products  
- Computed **null-safe aggregations** (mean ratings, variances, counts)  
- Applied **statistical imputations**: mean/median price, unknown title filling  

### Feature Engineering  
- **Text Features:** Trained **Word2Vec embeddings** (`vectorSize=16`, `minCount=100`) on product titles  
- **Categorical Features:** Encoded categories with **StringIndexer + OneHotEncoder**, then reduced dimensionality via **PCA (k=15)**  
- **Vector Summaries:** Used `M.stat.Summarizer.mean` to compute dense mean vectors for OHE and PCA outputs  

### Modeling  
- Trained **Decision Tree Regressors (PySpark ML)** for rating prediction  
- Hyperparameter tuning with depth sweep {5, 7, 9, 12} using seeded `randomSplit`  
- **Optimizations:**  
  - Persisted train/valid splits with `MEMORY_AND_DISK` to avoid recomputation  
  - Schema-pruned to only `features` and `overall` columns  
  - Explicit caching/unpersisting to manage shuffle-heavy workloads   

---

## 🛠️ Tech Stack  
- **Languages**: Python (PySpark)  
- **Libraries/Frameworks**: PySpark (SQL, MLlib DataFrame API), Spark ML  
- **Platforms/Tools**: UCSD DSMLP (Kubernetes-backed Spark cluster), Jupyter Notebook
- **Data**: ~12GB Amazon product + review dataset 
