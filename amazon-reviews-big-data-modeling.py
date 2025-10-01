import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
# import any other dependencies you want, but make sure only to use the ones
# availiable on AWS EMR
from pyspark.sql.functions import size
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark import StorageLevel

# ---------------- choose input format, dataframe or rdd ----------------------
INPUT_FORMAT = 'dataframe'  # change to 'rdd' if you wish to use rdd inputs
# -----------------------------------------------------------------------------
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics


# ---------- Begin definition of helper functions, if you need any ------------

# def task_1_helper():
#   pass

# -----------------------------------------------------------------------------


# %load -s task_1 assignment2.py
def task_1(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    agg_df = (review_data.groupBy(asin_column).agg(
        F.avg(overall_column).alias(mean_rating_column), 
        F.count(overall_column).alias(count_rating_column)
    ))

    product_df = product_data.select(asin_column)
    merged_df = product_df.join(agg_df, on=asin_column, how='left')

    stats = merged_df.agg(
        F.count("*").alias("count_total"),
        F.mean(mean_rating_column).alias("mean_meanRating"),
        F.variance(mean_rating_column).alias("variance_meanRating"),
        F.sum(F.when(F.col(mean_rating_column).isNull(), 1).otherwise(0)).alias("numNulls_meanRating"),
        F.mean(count_rating_column).alias("mean_countRating"),
        F.variance(count_rating_column).alias("variance_countRating"),
        F.sum(F.when(F.col(count_rating_column).isNull(), 1).otherwise(0)).alias("numNulls_countRating")
    ).collect()[0]

    count_total = stats["count_total"]
    mean_meanRating = stats["mean_meanRating"]
    variance_meanRating = stats["variance_meanRating"]
    numNulls_meanRating = stats["numNulls_meanRating"]
    mean_countRating = stats["mean_countRating"]
    variance_countRating = stats["variance_countRating"]
    numNulls_countRating = stats["numNulls_countRating"]

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Calculate the values programmaticly. Do not change the keys and do not
    # hard-code values in the dict. Your submission will be evaluated with
    # different inputs.
    # Modify the values of the following dictionary accordingly.
    res = {
        'count_total': None,
        'mean_meanRating': None,
        'variance_meanRating': None,
        'numNulls_meanRating': None,
        'mean_countRating': None,
        'variance_countRating': None,
        'numNulls_countRating': None
    }

    # Modify res:
    res['count_total'] = int(count_total)
    res['mean_meanRating'] = float(mean_meanRating)
    res['variance_meanRating'] = float(variance_meanRating)
    res['numNulls_meanRating'] = int(numNulls_meanRating)
    res['mean_countRating'] = float(mean_countRating)
    res['variance_countRating'] = float(variance_countRating)
    res['numNulls_countRating'] = int(numNulls_countRating)

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_1')
    return res
    # -------------------------------------------------------------------------


# %load -s task_2 assignment2.py
def task_2(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------
    
    category_df = product_data.withColumn(
        category_column,
        F.when(
            (F.col(categories_column).isNotNull()) &
            (size(F.col(categories_column)) > 0) &
            (size(F.col(categories_column).getItem(0)) > 0) &
            (F.col(categories_column).getItem(0).getItem(0) != ""),
            F.col(categories_column).getItem(0).getItem(0)
        ).otherwise(None)
    )
    
    final_df = category_df.withColumn(
        bestSalesCategory_column,
        F.map_keys(F.col(salesRank_column)).getItem(0)
    ).withColumn(
        bestSalesRank_column,
        F.map_values(F.col(salesRank_column)).getItem(0)
    )
    

    # Part 1
    count_total = final_df.count()

    # add columns that answer parts 2 and 3
    stats_sales = (
        final_df.agg(
            F.mean(bestSalesRank_column).alias("mean_bestSalesRank"),
            F.variance(bestSalesRank_column).alias("variance_bestSalesRank")
        )
        .collect()[0]
    )
    
    # Part 2-3
    mean_bestSalesRank = stats_sales["mean_bestSalesRank"]
    variance_bestSalesRank = stats_sales["variance_bestSalesRank"]

    # Part 4
    numNulls_category = final_df.filter(
        F.col(category_column).isNull()
    ).count()

    # Part 5
    countDistinct_category = (
        final_df.select(
            category_column
        ).where(
            F.col(category_column).isNotNull()
        ).distinct().count()
    )

    # Part 6
    numNulls_bestSalesCategory = final_df.filter(
        F.col(bestSalesCategory_column).isNull()
    ).count()

    # Part 7
    countDistinct_bestSalesCategory = (
        final_df.select(
            bestSalesCategory_column
        ).where(
            F.col(bestSalesCategory_column).isNotNull()
        ).distinct().count()
    )

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_bestSalesRank': None,
        'variance_bestSalesRank': None,
        'numNulls_category': None,
        'countDistinct_category': None,
        'numNulls_bestSalesCategory': None,
        'countDistinct_bestSalesCategory': None
    }
    # Modify res:
    res['count_total'] = int(count_total)
    res['mean_bestSalesRank'] = float(mean_bestSalesRank)
    res['variance_bestSalesRank'] = float(variance_bestSalesRank)
    res['numNulls_category'] = int(numNulls_category)
    res['countDistinct_category'] = int(countDistinct_category)
    res['numNulls_bestSalesCategory'] = int(numNulls_bestSalesCategory)
    res['countDistinct_bestSalesCategory'] = int(countDistinct_bestSalesCategory)


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_2')
    return res
    # -------------------------------------------------------------------------


# %load -s task_3 assignment2.py
def task_3(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    count_df = product_data.select(asin_column,
                                   F.col(related_column).getItem(attribute).alias("alsoViewedArray")
                                  ).withColumn(countAlsoViewed_column,
                                               F.when(
                                                   (F.col("alsoViewedArray").isNull()) | (F.size(F.col("alsoViewedArray")) == 0), None
                                               ).otherwise(F.size(F.col("alsoViewedArray")))
                                              ).select(asin_column, countAlsoViewed_column, "alsoViewedArray")

    exploded_df = count_df.select(asin_column, F.explode_outer("alsoViewedArray").alias("viewed_asin"))

    price_df = product_data.select(F.col(asin_column).alias("viewed_asin"), F.col(price_column).alias("viewed_price"))

    joined_df = exploded_df.join(price_df, on="viewed_asin", how="left").filter(
        F.col("viewed_price").isNotNull() & (F.col("viewed_price") > 0)
    ).select(asin_column, "viewed_price").cache()

    mean_df = joined_df.groupBy(asin_column).agg(F.mean("viewed_price").alias(meanPriceAlsoViewed_column))
    
    joined_df.unpersist()

    processed_df = count_df.select(asin_column, countAlsoViewed_column).join(mean_df, on=asin_column, how="left")

    final_df = product_data.select(asin_column).distinct().join(processed_df, on=asin_column, how="left")

    stats = final_df.agg(
        F.count("*").alias("count_total"),
        F.mean(meanPriceAlsoViewed_column).alias("mean_meanPriceAlsoViewed"),
        F.variance(meanPriceAlsoViewed_column).alias("variance_meanPriceAlsoViewed"),
        F.sum(F.when(F.col(meanPriceAlsoViewed_column).isNull(), 1).otherwise(0)).alias("numNulls_meanPriceAlsoViewed"),
        F.mean(countAlsoViewed_column).alias("mean_countAlsoViewed"),
        F.variance(countAlsoViewed_column).alias("variance_countAlsoViewed"),
        F.sum(F.when(F.col(countAlsoViewed_column).isNull(), 1).otherwise(0)).alias("numNulls_countAlsoViewed")
    ).collect()[0]
    
    count_total = stats["count_total"]
    mean_meanPriceAlsoViewed = stats["mean_meanPriceAlsoViewed"]
    variance_meanPriceAlsoViewed = stats["variance_meanPriceAlsoViewed"]
    numNulls_meanPriceAlsoViewed = stats["numNulls_meanPriceAlsoViewed"]
    mean_countAlsoViewed = stats["mean_countAlsoViewed"]
    variance_countAlsoViewed = stats["variance_countAlsoViewed"]
    numNulls_countAlsoViewed = stats["numNulls_countAlsoViewed"]


    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanPriceAlsoViewed': None,
        'variance_meanPriceAlsoViewed': None,
        'numNulls_meanPriceAlsoViewed': None,
        'mean_countAlsoViewed': None,
        'variance_countAlsoViewed': None,
        'numNulls_countAlsoViewed': None
    }

    # Modify res:
    res['count_total'] = int(count_total)
    res['mean_meanPriceAlsoViewed'] = float(mean_meanPriceAlsoViewed)
    res['variance_meanPriceAlsoViewed'] = float(variance_meanPriceAlsoViewed)
    res['numNulls_meanPriceAlsoViewed'] = int(numNulls_meanPriceAlsoViewed)
    res['mean_countAlsoViewed'] = float(mean_countAlsoViewed)
    res['variance_countAlsoViewed'] = float(variance_countAlsoViewed)
    res['numNulls_countAlsoViewed'] = int(numNulls_countAlsoViewed)

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_3')
    return res
    # -------------------------------------------------------------------------


# %load -s task_4 assignment2.py
def task_4(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    priceDouble_df = product_data.withColumn(
        "priceDouble",
        F.col(price_column).cast("double")
    )

    mean_price = (
        priceDouble_df.agg(
            F.mean("priceDouble").alias("mean_price")
        ).collect()[0]["mean_price"]
    )

    median_price = (
        priceDouble_df.select(
            F.expr("percentile_approx(priceDouble, 0.5)").alias("median_price")
        ).collect()[0]["median_price"]
    )

    meanImputed_df = priceDouble_df.withColumn(
        meanImputedPrice_column,
        F.when(
            F.col("priceDouble").isNull(), F.lit(mean_price)
        ).otherwise(F.col("priceDouble"))
    )

    medianImputed_df = meanImputed_df.withColumn(
        medianImputedPrice_column,
        F.when(
            F.col("priceDouble").isNull(), F.lit(median_price)
        ).otherwise(F.col("priceDouble"))
    )

    final_df = medianImputed_df.withColumn(
        unknownImputedTitle_column,
        F.when(
            F.col(title_column).isNull() | (F.col(title_column) == ""),
            F.lit("unknown")
        ).otherwise(F.col(title_column))
    )
    
    # Part 1
    count_total = final_df.count()

    # Part 2-3
    stats_mean_imp = (
        final_df.agg(
            F.mean(meanImputedPrice_column).alias("mean_meanImputedPrice"),
            F.variance(meanImputedPrice_column).alias("variance_meanImputedPrice")
        ).collect()[0]
    )
    mean_meanImputedPrice = stats_mean_imp["mean_meanImputedPrice"]
    variance_meanImputedPrice = stats_mean_imp["variance_meanImputedPrice"]

    # Part 4
    numNulls_meanImputedPrice = final_df.filter(
        F.col(meanImputedPrice_column).isNull()
    ).count()

    # Part 5-6
    stats_median_imp = (
        final_df.agg(
            F.mean(medianImputedPrice_column).alias("mean_medianImputedPrice"),
            F.variance(medianImputedPrice_column).alias("variance_medianImputedPrice")
        ).collect()[0]
    )
    mean_medianImputedPrice = stats_median_imp["mean_medianImputedPrice"]
    variance_medianImputedPrice = stats_median_imp["variance_medianImputedPrice"]

    # Part 7
    numNulls_medianImputedPrice = final_df.filter(
        F.col(medianImputedPrice_column).isNull()
    ).count()

    # Part 8
    numUnknowns_unknownImputedTitle = final_df.filter(
        F.col(unknownImputedTitle_column) == "unknown"
    ).count()

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None, 
        'mean_meanImputedPrice': None,
        'variance_meanImputedPrice': None,
        'numNulls_meanImputedPrice': None,
        'mean_medianImputedPrice': None,
        'variance_medianImputedPrice': None,
        'numNulls_medianImputedPrice': None,
        'numUnknowns_unknownImputedTitle': None
    }
    # Modify res:
    res['count_total'] = int(count_total)
    res['mean_meanImputedPrice'] = float(mean_meanImputedPrice)
    res['variance_meanImputedPrice'] = float(variance_meanImputedPrice)
    res['numNulls_meanImputedPrice'] = int(numNulls_meanImputedPrice)
    res['mean_medianImputedPrice'] = float(mean_medianImputedPrice)
    res['variance_medianImputedPrice'] = float(variance_medianImputedPrice)
    res['numNulls_medianImputedPrice'] = int(numNulls_medianImputedPrice)
    res['numUnknowns_unknownImputedTitle'] = int(numUnknowns_unknownImputedTitle)


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_4')
    return res
    # -------------------------------------------------------------------------


# %load -s task_5 assignment2.py
def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    # -------------------------------------------------------------------------

    # ---------------------- Your implementation begins------------------------

    product_processed_data_output = product_processed_data.select(title_column).withColumn(titleArray_column, F.split(F.lower(F.col(title_column)), ' ')).cache()

    w2v = M.feature.Word2Vec(inputCol=titleArray_column, outputCol=titleVector_column, vectorSize=16, minCount=100, seed=SEED, numPartitions=4)

    model = w2v.fit(product_processed_data_output)

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'size_vocabulary': None,
        'word_0_synonyms': [(None, None), ],
        'word_1_synonyms': [(None, None), ],
        'word_2_synonyms': [(None, None), ]
    }
    # Modify res:
    res['count_total'] = product_processed_data_output.count()
    res['size_vocabulary'] = model.getVectors().count()
    for name, word in zip(
        ['word_0_synonyms', 'word_1_synonyms', 'word_2_synonyms'],
        [word_0, word_1, word_2]
    ):
        res[name] = model.findSynonymsArray(word, 10)
    
    product_processed_data_output.unpersist()
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_5')
    return res
    # -------------------------------------------------------------------------


# %load -s task_6 assignment2.py
def task_6(data_io, product_processed_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'
    # -------------------------------------------------------------------------    

    # ---------------------- Your implementation begins------------------------
    product_processed_data = product_processed_data.select(category_column)

    indexer = M.feature.StringIndexer(inputCol=category_column, outputCol=categoryIndex_column, handleInvalid='keep')
    indexed_data = indexer.fit(product_processed_data).transform(product_processed_data)

    ohe = M.feature.OneHotEncoder(inputCol=categoryIndex_column, outputCol=categoryOneHot_column, dropLast=True)
    ohe_data = ohe.fit(indexed_data).transform(indexed_data)

    pca = M.feature.PCA(k=15, inputCol=categoryOneHot_column, outputCol=categoryPCA_column)
    pca_data = pca.fit(ohe_data).transform(ohe_data)

    ohe_mean_vector = pca_data.agg(M.stat.Summarizer.mean(F.col(categoryOneHot_column)).alias("mean")).first()["mean"].toArray().tolist()
    pca_mean_vector = pca_data.agg(M.stat.Summarizer.mean(F.col(categoryPCA_column)).alias("mean")).first()["mean"].toArray().tolist()

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'meanVector_categoryOneHot': [None, ],
        'meanVector_categoryPCA': [None, ]
    }
    # Modify res:
    res['count_total'] = pca_data.count()
    res['meanVector_categoryOneHot'] = ohe_mean_vector
    res['meanVector_categoryPCA'] = pca_mean_vector

    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_6')
    return res
    # -------------------------------------------------------------------------

    
    
def task_7(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------

    train_data = train_data.select("features", "overall")
    test_data = test_data.select("features", "overall")

    model = DecisionTreeRegressor(featuresCol="features", labelCol="overall", maxDepth=5).fit(train_data)
    
    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="overall", predictionCol="prediction", metricName="rmse")
    test_rmse = evaluator.evaluate(predictions)
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None
    }
    # Modify res:
    res['test_rmse'] = float(test_rmse)


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_7')
    return res
    # -------------------------------------------------------------------------
    
    
def task_8(data_io, train_data, test_data):
    
    # ---------------------- Your implementation begins------------------------
    
    train_data = train_data.select("features", "overall")
    test_data = test_data.select("features", "overall")

    train_split, valid_split = train_data.randomSplit([0.75, 0.25], seed=SEED)
    train_split = train_split.persist(StorageLevel.MEMORY_AND_DISK)
    valid_split = valid_split.persist(StorageLevel.MEMORY_AND_DISK)

    evaluator = RegressionEvaluator(labelCol="overall", predictionCol="prediction", metricName="rmse")
    depths = [5, 7, 9, 12]


    val_rmses = {}
    best_model = None
    best_rmse = float('inf')
    best_depth = None

    for d in depths:
        model = DecisionTreeRegressor(featuresCol="features", labelCol="overall", maxDepth=d).fit(train_split)
        predictions = model.transform(valid_split)
        rmse = evaluator.evaluate(predictions)
        val_rmses[d] = float(rmse)

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_depth = d

    train_split.unpersist()
    valid_split.unpersist()


    test_predictions = best_model.transform(test_data)
    test_rmse = evaluator.evaluate(test_predictions)
    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None,
        'valid_rmse_depth_5': None,
        'valid_rmse_depth_7': None,
        'valid_rmse_depth_9': None,
        'valid_rmse_depth_12': None
    }
    # Modify res:
    res['test_rmse'] = float(test_rmse)
    res['valid_rmse_depth_5'] = val_rmses[5]
    res['valid_rmse_depth_7'] = val_rmses[7]
    res['valid_rmse_depth_9'] = val_rmses[9]
    res['valid_rmse_depth_12'] = val_rmses[12]

    
    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_8')
    return res
    # -------------------------------------------------------------------------