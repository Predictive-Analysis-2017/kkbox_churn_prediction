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
import scala.util.Random

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
  
    def cityMappint(city: Int): Int = {
      city
    } 
    def regMapping(reg: Int): Int = {
      if(reg == 9){
        0
      }else if(reg == 3){
        1
      }else if(reg == 4){
        2
      }else if(reg == 7){
        3
      }else if(reg == 16){
        4
      }else if(reg == 13){
        5
      }else if(reg == 10){
        6
      }else{
        7
      }
    }

    val cityRdd = membersRddClean1.map{line => val data = line.split(",")
      data(1)
    }.distinct()
    println("city count: " + cityRdd.count())
    cityRdd.take(25).foreach(println)
    val regRdd = membersRddClean1.map{line => val data = line.split(",")
      data(4)
    }.distinct()
    println("reg count: " + regRdd.count())
    regRdd.take(25).foreach(println)


    // map member data to (msno, city, bd, register_via)
    val FeatureRdd1 = membersRddClean1.map{line => val data = line.split(",")
      val city = data(1)
      var sparse_city = Array.fill[Double](22)(0.0)
      sparse_city(city.toInt - 1) = 1.0
      var reg = data(4)
      var sparse_reg = Array.fill[Double](8)(0.0)
      sparse_reg(regMapping(reg.toInt)) = 1.0 
       
      //(data(0), (data(1), data(2), data(4)))
      (data(0), (sparse_city, data(2), sparse_reg))
    }

    // map train data to (msno, is_churn)
    val FeatureRdd2 = trainRddClean1.map{line => val data = line.split(",")
      (data(0), (data(1)))
    }

    //use msno to join tables
    val combinedFeatures = FeatureRdd1.join(FeatureRdd2)
    var combinedFeaturesCount = combinedFeatures.count()
    println("combined count = " + combinedFeaturesCount)

    val churnRdd = combinedFeatures.filter(line => line._2._2.toInt == 1).count()
    val notChurnRdd = combinedFeatures.filter(line => line._2._2.toInt == 0).count()
    println("Churn count: " + churnRdd)
    println("NOT Churn count: " + notChurnRdd)

    //since the data is so unbalanced, we need do resampling
    val combinedFeaturesEvenSize = combinedFeatures.map{line => val data = line
      val rg = new scala.util.Random
      val f = rg.nextDouble
      (line, f)
    }.filter{line => val data = line
      if(data._1._2._2.toInt == 1){
        true
      }else{
        if(data._2 >= 0.07){
          false
        }else{
          true
        }
      } 
    }.map(line => line._1)

    val churnRdd2 = combinedFeaturesEvenSize.filter(line => line._2._2.toInt == 1).count()
    val notChurnRdd2 = combinedFeaturesEvenSize.filter(line => line._2._2.toInt == 0).count()
    println("Churn count: " + churnRdd2)
    println("NOT Churn count: " + notChurnRdd2)

    //it's very stupid, should find how to write this elegantly...
    var data = combinedFeaturesEvenSize.map{line => val data = line
      //val c = Vectors.dense(data._2._1._1.toDouble, data._2._1._2.toDouble, data._2._1._3.toDouble)
      val c = Vectors.dense(data._2._1._1(0), data._2._1._1(1), data._2._1._1(2), data._2._1._1(3), data._2._1._1(4), data._2._1._1(5), data._2._1._1(6), data._2._1._1(7), data._2._1._1(8), data._2._1._1(9), data._2._1._1(10), data._2._1._1(11), data._2._1._1(12), data._2._1._1(13), data._2._1._1(14), data._2._1._1(15), data._2._1._1(16), data._2._1._1(17), data._2._1._1(18), data._2._1._1(19), data._2._1._1(20), data._2._1._1(21), data._2._1._2.toDouble, data._2._1._3(0), data._2._1._3(1), data._2._1._3(2), data._2._1._3(3), data._2._1._3(4), data._2._1._3(5), data._2._1._3(6), data._2._1._3(7))
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
