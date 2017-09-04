# This code only works against HDI:
# 1. Download the AirOnTimeCSV.zip file from http://packages.revolutionanalytics.com/datasets/AirOnTime87to12/
# 2. Expand the zip file into 303 csv files, then upload them to a container named "airontime" in the storage account where the HDI has a default container mapped to it.
# 3. Configure a compute context against the HDI cluster.
# 4. Run the code against that HDI cluster.

import pyspark
import pyspark.ml

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.functions import col
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import *
from pyspark.ml.feature import OneHotEncoder, StringIndexer

class timeit():
    from datetime import datetime
    def __enter__(self):
        self.tic = self.datetime.now()
    def __exit__(self, *args, **kwargs):
        print('Total running time: {}'.format(self.datetime.now() - self.tic))

with timeit():   
    # Start Spark application
    spark = pyspark.sql.SparkSession.builder.appName("Air On Time").getOrCreate()

    # Load all .csv files in airontime folder
    air = spark.read.csv('wasb:///airontime/*.csv', header=True)    

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
        
    # string index and one-hot encode DAY_OF_WEEK column
    siDoW = StringIndexer(inputCol="DAY_OF_WEEK", outputCol="doWIndex")
    ecDow = OneHotEncoder(inputCol=siDoW.getOutputCol(), outputCol="e1")

    # string index and one-hot encode MONTH column
    siM = StringIndexer(inputCol='MONTH', outputCol='monthIndex')
    ecM = OneHotEncoder(inputCol=siM.getOutputCol(), outputCol="e2")

    # string index and one-hot encode UNIQUE_CARRIER column
    siCarrier = StringIndexer(inputCol='UNIQUE_CARRIER', outputCol='carrierIndex')
    ecC = OneHotEncoder(inputCol=siCarrier.getOutputCol(), outputCol='ca')

    # assemble vectors
    assembler = VectorAssembler(inputCols=[ecDow.getOutputCol(), ecM.getOutputCol(), ecC.getOutputCol()], outputCol="features")

    # create a Logistic Regression model
    lr = LogisticRegression()

    # create the pipeline
    pipe = Pipeline(stages=[siDoW, siM, ecDow, ecM, siCarrier, ecC, assembler, lr])

    # train
    model = pipe.fit(train)

    # score
    pred = model.transform(test)

    # evaluate
    bce = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
    auc = bce.setMetricName('areaUnderROC').evaluate(pred)

    # print AUC
    print('AUC: {}'.format(auc))
