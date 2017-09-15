import numpy as np
import pandas as pd
import mmlspark

import pyspark
import pyspark.ml

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col

from mmlspark.TrainClassifier import TrainClassifier
from pyspark.ml.classification import LogisticRegression

class timeit():
    from datetime import datetime
    def __enter__(self):
        self.tic = self.datetime.now()
    def __exit__(self, *args, **kwargs):
        print('Total running time: {}'.format(self.datetime.now() - self.tic))

with timeit():   
    # Start Spark application
    spark = pyspark.sql.SparkSession.builder.appName("Air On Time from Hai").getOrCreate()

    # Load all .csv files in airontime folder
    air = spark.read.csv('wasb:///airontime/*.csv', header=True).sample(True, 0.001, seed=42)    

    # cache and count
    print("Total number of rows: {}".format(air.cache().count()))
    
    # remote missing values and select only 6 columsn out of 47
    data = air\
        .where('ARR_DEL15 IS NOT NULL AND CRS_ELAPSED_TIME IS NOT NULL')\
        .select('MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER', 'DISTANCE', 'CRS_ELAPSED_TIME', 'ARR_DEL15')

    # set correct datat types
    data = data.withColumn('CRS_ELAPSED_TIME', data['CRS_ELAPSED_TIME'].cast(DoubleType()))
    data = data.withColumn('DISTANCE', data['DISTANCE'].cast(DoubleType()))
    data = data.withColumn('label', data['ARR_DEL15'].cast(DoubleType()))
    print("Total number of rows after wrangling: {}".format(data.cache().count()))
    data.printSchema()
    
    # split data to training and test sets
    train, test = data.select('DAY_OF_WEEK', 'MONTH', 'DISTANCE', 'UNIQUE_CARRIER', 'label').randomSplit([0.7, 0.3], seed=123)
    print('Number of rows in training dataset: {}'.format(train.cache().count()))
    print('Number of rows in test data size: {}'.format(test.cache().count()))
        
    # train
    model = TrainClassifier(model=LogisticRegression(), labelCol="label", numFeatures=256).fit(train)

    # score
    from mmlspark.ComputeModelStatistics import ComputeModelStatistics
    prediction = model.transform(test)
    metrics = ComputeModelStatistics().transform(prediction)
    print(metrics.limit(10).toPandas())
    
    # save model
    model.write().overwrite().save("./outputs/aot-mmlspark.model")
