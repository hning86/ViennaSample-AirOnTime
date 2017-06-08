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
from azureml_sdk import data_collector

# start Spark session
spark = pyspark.sql.SparkSession.builder.appName('AirOnTime').getOrCreate()
# instantiate a logger
run_logger = data_collector.current_run() 

# read csv folder from attached wasb
air = spark.read.csv('wasb:///airontime/*.csv', header=True)

# take a very small sample
#data = air.sample(False, 0.00001, seed=123).where('ARR_DEL15 IS NOT NULL').select('MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER', 'CRS_ELAPSED_TIME', 'ARR_DEL15')

# load data from a local parquet file
#air = spark.read.parquet('AirOnTime_sample_15k.parquet')

# select a list of relevant columns
data = air.where('ARR_DEL15 IS NOT NULL AND CRS_ELAPSED_TIME IS NOT NULL').select('MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER', 'CRS_ELAPSED_TIME', 'ARR_DEL15')

# convert CRS_ELAPSED_TIME to double.
data = data.withColumn('CRS_ELAPSED_TIME', data['CRS_ELAPSED_TIME'].cast(DoubleType()))

# rename "ARR_DEL15" column to "label", and cast it to double.
data = data.withColumn('label', data['ARR_DEL15'].cast(DoubleType()))

# cache the raw dataset
numOfRows = data.cache().count()
print('Total number of rows: {}'.format(numOfRows))

# split training and test datasets
train, test = data.select('DAY_OF_WEEK', 'MONTH', 'UNIQUE_CARRIER', 'CRS_ELAPSED_TIME', 'label').randomSplit([0.75, 0.25])

# cache the training set
print('train data size: {}'.format(train.cache().count()))

# cache the test set
print('test data size: {}'.format(test.cache().count()))

# string index and one-hot encode DAY_OF_WEEK column
siDoW = StringIndexer(inputCol="DAY_OF_WEEK", outputCol="doWIndex")
ecDow = OneHotEncoder(inputCol=siDoW.getOutputCol(), outputCol="e1")

# string index and one-hot encode MONTH column
siM = StringIndexer(inputCol='MONTH', outputCol='monthIndex')
ecM = OneHotEncoder(inputCol=siM.getOutputCol(), outputCol="e2")

# string index and one-hot encode UNIQUE_CARRIER column
siCarrier = StringIndexer(inputCol='UNIQUE_CARRIER', outputCol='carrierIndex')
ecC = OneHotEncoder(inputCol=siCarrier.getOutputCol(), outputCol='ca')

# assemble numeric features into a single vector column named 'features'
assembler = VectorAssembler(inputCols=['e1', 'e2', 'ca', 'CRS_ELAPSED_TIME'], outputCol="features")

# Logistic Regression algorithm with default parameter settings
lr = LogisticRegression(maxIter=50, regParam=0.01, elasticNetParam=0.8)

# create the pipeline
pipe = Pipeline(stages=[siDoW, siM, ecDow, ecM, siCarrier, ecC, assembler, lr])

# train
model = pipe.fit(train)

# score
pred = model.transform(test)

# evaluate
bce = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
auc = bce.setMetricName('areaUnderROC').evaluate(pred)

print()
print('######################################')
print('AUC: {}'.format(auc))
print('######################################')

# log AUC
run_logger.log("AUC", auc)

# save the model
model.write().overwrite().save("wasb:///models/AirOnTimeModel.mml")