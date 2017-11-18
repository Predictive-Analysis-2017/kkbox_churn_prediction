import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import math._
import java.util.Calendar
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.functions.rand

object pa_final {
  def main(args: Array[String]) {
    val sc = new SparkContext(new SparkConf().setAppName("PA Final"))

    /* JUST FOR TESTING */

    //train
    val trainRdd = sc.textFile("hdfs:/user/hungwei/train.csv")
    val trainRddHeader = trainRdd.first()
    val trainRddClean1 = trainRdd.filter(line => line != trainRddHeader)

    //transaction
    val transactionRdd = sc.textFile("hdfs:/user/hungwei/transactions.csv")
    val transactionRddHeader = transactionRdd.first()
    val transactionRddClean1 = transactionRdd.filter(line => line != transactionRddHeader)

    //user_logs
    val userLogsRdd = sc.textFile("hdfs:/user/hungwei/user_logs.csv")
    val userLogsRddHeader = userLogsRdd.first()
    val userLogsRddClean1 = userLogsRdd.filter(line => line != userLogsRddHeader)

    //members
    val membersRdd = sc.textFile("hdfs:/user/hungwei/members.csv")
    val membersRddHeader = membersRdd.first()
    val membersRddClean1 = membersRdd.filter(line => line != membersRddHeader)

    //check how many users' age are valid
    val membersRddClean3 = membersRddClean1.filter{line => val data = line.split(",")
      if(data(2) == null || data(2).toInt < 3 || data(2).toInt > 120)
        false
      else
        true
    }

    val membersRddMale = membersRddClean1.filter{line => val data = line.split(",")
      if(data(3) == "male")
        true
      else
        false
    }
    val membersRddFemale = membersRddClean1.filter{line => val data = line.split(",")
      if(data(3) == "female")
        true
      else
        false
    }

    val maleAvgAge = membersRddClean3.filter{line => val data = line.split(",")
      if(data(3) == "male")
        true
      else
        false
    }.map(line => line.split(",")(2).toDouble)
    val maleAge = maleAvgAge.reduce(_ + _) / maleAvgAge.count()

    val femaleAvgAge = membersRddClean3.filter{line => val data = line.split(",")
      if(data(3) == "female")
        true
      else
        false
    }.map(line => line.split(",")(2).toDouble)
    val femaleAge = femaleAvgAge.reduce(_ + _) / femaleAvgAge.count()


    //print information
    var trainRddCount = trainRddClean1.count()
    println("train rdd count = " + trainRddCount)
    var transactionRddCount = transactionRddClean1.count()
    println("transaction rdd count = " + transactionRddCount)
    var userLogsRddCount = userLogsRddClean1.count()
    println("user logs rdd count = " + userLogsRddCount)
    var membersRddCount = membersRddClean1.count()
    println("members rdd count = " + membersRddCount)
    var membersRddCountValid = membersRddClean3.count()
    println("members rdd count after removing outlier - age = " + membersRddCountValid)
    var membersRddCountNotValid = membersRddCount - membersRddCountValid
    println("members rdd count - not valid = " + membersRddCountNotValid)
    var maleCount = membersRddMale.count()
    var femaleCount = membersRddFemale.count()
    var noGenderCount = membersRddCount - maleCount - femaleCount
    println("male members rdd count = " + maleCount)
    println("female members rdd count = " + femaleCount)
    println("members without gender count = " + noGenderCount)
    println("male avg age = " + maleAge)
    println("female avg age = " + femaleAge)

    /* FINISH TESTING */

    /* PREPARE DATA */

    // map member data to (msno, city, bd, register_via)
    val FeatureRdd1 = membersRddClean1.map{line => val data = line.split(",")
      (data(0), (data(1), data(2), data(4)))
    } 
    // map train data to (msno, is_churn)
    val FeatureRdd2 = trainRddClean1.map{line => val data = line.split(",")
      (data(0), (data(1)))
    }

    val combinedFeatures = FeatureRdd1.join(FeatureRdd2)
    var combinedFeaturesCount = combinedFeatures.count()
    println("combined count = " + combinedFeaturesCount)

    val churnRdd = combinedFeatures.filter(line => line._2._2.toInt == 1).count()
    val notChurnRdd = combinedFeatures.filter(line => line._2._2.toInt == 0).count()
    println("Churn count: " + churnRdd);
    println("NOT Churn count: " + notChurnRdd);

    var data = combinedFeatures.map{line => val data = line
      val c = Vectors.dense(data._2._1._1.toDouble, data._2._1._2.toDouble, data._2._1._3.toDouble)
      var y = data._2._2.toInt
      LabeledPoint(y, c)
    }.cache()
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val Array(trainData, testData) = data.randomSplit(Array(0.7, 0.3))
    val pointsTrainDf = sqlContext.createDataFrame(trainData)
    val pointsTrainDs = pointsTrainDf.as[LabeledPoint]
    val pointsTestDf = sqlContext.createDataFrame(testData)
    val pointsTestDs = pointsTestDf.as[LabeledPoint]
    val pointsDf = sqlContext.createDataFrame(data)
    val pointsDs = pointsDf.as[LabeledPoint]

    var nPTs = pointsTestDs.count()
    println("nPTs = " + nPTs)
    var nPDs = pointsDs.count()
    println("nPDs = " + nPDs)

    /* FINISH PREPARING DATA */

    /* DECISION TREE */
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(pointsDf)
    // Automatically identify categorical features, and index them.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(10) // features with > 4 distinct values are treated as continuous.
      .fit(pointsDf)

    // Split the data into training and test sets (30% held out for testing).
    //val Array(trainingData, testData) = pointsTrainDs.randomSplit(Array(0.7, 0.3))

    // Train a DecisionTree model.
    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(pointsTrainDf)

    // Make predictions.
    val predictions = model.transform(pointsTestDf)

    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(50)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model:\n" + treeModel.toDebugString)

    /* FINISH RUNNING DECISION TREE */
  
    sc.stop()
  }
}
