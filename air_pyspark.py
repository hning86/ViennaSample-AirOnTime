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

#data = air.sample(False, 0.00001, seed=123).where('ARR_DEL15 IS NOT NULL').select('MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER', 'ARR_DEL15')
data = air.where('ARR_DEL15 IS NOT NULL').select('MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER', 'ARR_DEL15')
data = data.withColumn('MONTH', data['MONTH'].cast(DoubleType()))
data = data.withColumn('DAY_OF_WEEK', data['DAY_OF_WEEK'].cast(DoubleType()))
# rename "ARR_DEL15" column to "label"
data = data.withColumn('label', data['ARR_DEL15'].cast(DoubleType()))

# cache the raw dataset
numOfRows = data.cache().count()
print('Total number of rows: {}'.format(numOfRows))

train, test = data.select('DAY_OF_WEEK', 'MONTH', 'UNIQUE_CARRIER', 'label').randomSplit([0.8, 0.2])
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

# assemble vectors
assembler = VectorAssembler(inputCols=[ecDow.getOutputCol(), ecM.getOutputCol(), ecC.getOutputCol()], outputCol="features")

# Logistic Regression algorithm with default parameter settings
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

print()
print('######################################')
print('AUC: {}'.format(auc))
print('######################################')

run_logger.log("AUC", auc)
