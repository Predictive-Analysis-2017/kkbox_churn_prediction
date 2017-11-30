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
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.functions.rand
import scala.util.Random
import scala.collection.mutable.ListBuffer
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

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

    //transaction special features
    val tranSpecial1Rdd = sc.textFile("hdfs:/user/hungwei/feature/part-*")
    val tranSpecial1Rdd1 = tranSpecial1Rdd.map{line => val data = line
      val t = data.split(",")
      (t(0), (t(1), t(2), t(3), t(4), t(5), t(6), t(7), t(8), t(9)))
    }

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
    val tranRdd = transactionRddClean1.map{line => val data = line.split(",")
      data(1)
    }.distinct()
    println("tran count: " + tranRdd.count())
    tranRdd.take(25).foreach(println)

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

    // map transaction data to (msno, payment_method_id, payment_plan_days, plan_list_price, is_auto_new, is_cancel, month of subscription
    val FeatureRdd3 = transactionRddClean1.map{line => val data = line.split(",")
      val date = data(6)
      val month = date.slice(4, 6)
      val method = data(1)
      var sparse_method = Array.fill[Double](42)(0.0)
      sparse_method(method.toInt - 1) = 1.0
      var sparse_month = Array.fill[Double](12)(0.0)
      sparse_month(month.toInt - 1) = 1.0

      (data(0), (sparse_method, data(2), data(3), data(5), data(8), sparse_month, data(6)))
    }

    // only choose 1 transaction data as our feature
    val FeatureRdd3_Unique = FeatureRdd3.reduceByKey{(x, y) => 
      if(x._7 > y._7){
        x
      }else{
        y
      }
    }

    // map user_logs data to (msno, avg_p25, avg_p50, avg_p75, avg_p985, avg_p100, avg_unique, avg_total_second)
    val FeatureRdd4 = userLogsRddClean1.map{line => val data = line.split(",")
      (data(0), (data(2).toDouble, data(3).toDouble, data(4).toDouble, data(5).toDouble, data(6).toDouble, data(7).toDouble, data(8).toDouble))
    }
    val FeatureRdd4_1 = FeatureRdd4.mapValues(line => (line, 1.0))
    val FeatureRdd4_2 = FeatureRdd4_1.reduceByKey{(x, y) => 
      ((x._1._1 + y._1._1, x._1._2 + y._1._2, x._1._3 + y._1._3, x._1._4 + y._1._4, x._1._5 + y._1._5, x._1._6 + y._1._6, x._1._7 + y._1._7), x._2 + y._2)
    }
    // avg features seems useless, only add total seconds played and log counts into our features
    val FeatureRdd4_3 = FeatureRdd4_2.map{line => val data = line
      //(data._1, (data._2._1._1 / data._2._2, data._2._1._2 / data._2._2, data._2._1._3 / data._2._2, data._2._1._4 / data._2._2, data._2._1._5 / data._2._2, data._2._1._6 / data._2._2, data._2._1._7 / data._2._2, data._2._1._7, data._2._2))
      (data._1, (data._2._1._7, data._2._2))
    }

    val combinedFeatures0 = FeatureRdd2.join(FeatureRdd1)
    val combinedFeatures1 = combinedFeatures0.map{line => val data = line
      (data._1, (data._2._1, data._2._2._1, data._2._2._2, data._2._2._3))
    }
    val combinedFeatures2 = combinedFeatures1.join(FeatureRdd3_Unique)
    val combinedFeatures3 = combinedFeatures2.map{line => val data = line
      (data._1, (data._2._1._1, data._2._1._2, data._2._1._3, data._2._1._4, data._2._2._1, data._2._2._2, data._2._2._3, data._2._2._4, data._2._2._5, data._2._2._6))
    }

    val combinedFeatures4 = combinedFeatures3.join(tranSpecial1Rdd1)
    val combinedFeatures5 = combinedFeatures4.map{line => val data = line
      var sparse = Array.fill[Double](9)(0.0)
      sparse(0) = data._2._2._1.toDouble
      sparse(1) = data._2._2._2.toDouble
      sparse(2) = data._2._2._3.toDouble
      sparse(3) = data._2._2._4.toDouble
      sparse(4) = data._2._2._5.toDouble
      sparse(5) = data._2._2._6.toDouble
      sparse(6) = data._2._2._7.toDouble
      sparse(7) = data._2._2._8.toDouble
      sparse(8) = data._2._2._9.toDouble

      (data._1, (data._2._1._1, data._2._1._2, data._2._1._3, data._2._1._4, data._2._1._5, data._2._1._6, data._2._1._7, data._2._1._8,data._2._1._9, data._2._1._10, sparse))
    }

    val combinedFeatures6 = combinedFeatures5.join(FeatureRdd4_3)
    val combinedFeatures7 = combinedFeatures6.map{line => val data = line
      var sparse = Array.fill[Double](7)(0.0)
      sparse(0) = data._2._2._1.toDouble
      sparse(1) = data._2._2._2.toDouble
      /*sparse(2) = data._2._2._3.toDouble
      sparse(3) = data._2._2._4.toDouble
      sparse(4) = data._2._2._5.toDouble
      sparse(5) = data._2._2._6.toDouble
      sparse(6) = data._2._2._7.toDouble
      sparse(7) = data._2._2._8.toDouble
      sparse(8) = data._2._2._9.toDouble*/
   
      (data._1, (data._2._1._1, data._2._1._2, data._2._1._3, data._2._1._4, data._2._1._5, data._2._1._6, data._2._1._7, data._2._1._8,data._2._1._9, data._2._1._10, data._2._1._11, sparse))
    }

    var combinedFeaturesCount = combinedFeatures7.count()
    println("combined count = " + combinedFeaturesCount)

    val combinedFeatures = combinedFeatures7
    combinedFeatures.take(5).foreach(println)

    val churnRdd = combinedFeatures.filter(line => line._2._1.toInt == 1).count()
    val notChurnRdd = combinedFeatures.filter(line => line._2._1.toInt == 0).count()
    println("Churn count: " + churnRdd)
    println("NOT Churn count: " + notChurnRdd)

    //since the data is so unbalanced, we need do resampling
    val combinedFeaturesEvenSize = combinedFeatures.map{line => val data = line
      val rg = new scala.util.Random
      val f = rg.nextDouble
      (line, f)
    }.filter{line => val data = line
      if(data._1._2._1.toInt == 1){
        true
      }else{
        if(data._2 >= 0.07){
          false
        }else{
          true
        }
      } 
    }.map(line => line._1)

    val churnRdd2 = combinedFeaturesEvenSize.filter(line => line._2._1.toInt == 1).count()
    val notChurnRdd2 = combinedFeaturesEvenSize.filter(line => line._2._1.toInt == 0).count()
    println("Churn count: " + churnRdd2)
    println("NOT Churn count: " + notChurnRdd2)

    var data = combinedFeaturesEvenSize.map{line => val data = line
      //feature, count, accumulated count
      val v1 = Vectors.dense(data._2._2.toArray) //city 22, 22
      val v2 = Vectors.dense(data._2._3.toDouble)//bd 1, 23
      val v3 = Vectors.dense(data._2._4.toArray) //register_via 8, 31
      val v4 = Vectors.dense(data._2._5.toArray) //payment_method 42, 73
      val v5 = Vectors.dense(data._2._6.toDouble) //payment_plan_days 1, 74
      val v6 = Vectors.dense(data._2._7.toDouble) //plan_list_price 1, 75
      val v7 = Vectors.dense(data._2._8.toDouble) //is_auto_renew 1, 76
      val v8 = Vectors.dense(data._2._9.toDouble) //is_cancel 1, 77
      val v9 = Vectors.dense(data._2._10.toArray) //month 12, 89
      val v10 = Vectors.dense(data._2._11.toArray) //special features1 9, 98 
      val v11 = Vectors.dense(data._2._12.toArray) //user logs features 9, 105

      var y = data._2._1.toInt
      val features = Vectors.dense(v1.toArray ++ v2.toArray ++ v3.toArray ++ v4.toArray ++ v5.toArray ++ v6.toArray ++ v7.toArray ++ v8.toArray ++ v9.toArray ++ v10.toArray ++ v11.toArray)
      LabeledPoint(y, features)
    }.cache()
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(data.map(x => x.features))
    val scaledData = data.map(x =>
      LabeledPoint(x.label, scaler.transform(Vectors.dense(x.features.toArray)))).cache()
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    //val Array(trainData, testData) = data.randomSplit(Array(0.7, 0.3))
    val Array(trainData, testData) = scaledData.randomSplit(Array(0.7, 0.3))
    val pointsTrainDf = sqlContext.createDataFrame(trainData)
    val pointsTrainDs = pointsTrainDf.as[LabeledPoint]
    val pointsTestDf = sqlContext.createDataFrame(testData)
    val pointsTestDs = pointsTestDf.as[LabeledPoint]
    //val pointsDf = sqlContext.createDataFrame(data)
    val pointsDf = sqlContext.createDataFrame(scaledData)
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
      .setMaxDepth(5)//default value is 5

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
   
    val predictionsTrain = model.transform(pointsTrainDf)
    val evaluatorTrain = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    val accuracyTrain = evaluatorTrain.evaluate(predictionsTrain)
    println("Train Error = " + (1.0 - accuracyTrain))


    // Make predictions.
    val predictions = model.transform(pointsTestDf)

    // Select example rows to display.
    //predictions.select("predictedLabel", "label", "features").show(50)

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
    /* RANDOM FOREST */
    
    val rfLabelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(pointsDf)
    val rfFeatureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(pointsDf)

    // Train a DecisionTree model.
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)

    // Convert indexed labels back to original labels.
    val rfLabelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(rfLabelIndexer.labels)

    // Chain indexers and tree in a Pipeline.
    val rfPipeline = new Pipeline()
      .setStages(Array(rfLabelIndexer, rfFeatureIndexer, rf, rfLabelConverter))

    // Train model. This also runs the indexers.
    val rfModel = rfPipeline.fit(pointsTrainDf)
   
    val rfPredictionsTrain = rfModel.transform(pointsTrainDf)
    val rfEvaluatorTrain = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    val rfAccuracyTrain = rfEvaluatorTrain.evaluate(rfPredictionsTrain)
    println("RF Train Error = " + (1.0 - rfAccuracyTrain))


    // Make predictions.
    val rfPredictions = rfModel.transform(pointsTestDf)

    // Select (prediction, true label) and compute test error.
    val rfEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")
    val rfAccuracy = rfEvaluator.evaluate(rfPredictions)
    println("RF Test Error = " + (1.0 - rfAccuracy))
    val rfDrawModel = rfModel.stages(2).asInstanceOf[RandomForestClassificationModel]
    println("Learned classification forest model:\n" + rfDrawModel.toDebugString)
    
    /* FINISH RUNNING RANDOM FOREST */

    /* SVM */
    val svmNumIterations = 200
    val svmModel = SVMWithSGD.train(trainData, svmNumIterations)
    svmModel.clearThreshold()
    val svmScoreAndLabels = testData.map{point =>
      val score = svmModel.predict(point.features)
      (score, point.label)
    }
    svmScoreAndLabels.take(10).foreach(println)
    val metrics = new BinaryClassificationMetrics(svmScoreAndLabels)
    val auROC = metrics.areaUnderROC()
    println("SVM area under ROC = " + auROC)
    val svmAcc = svmScoreAndLabels.filter{line => val x = line
      if((x._1 >= 0 && x._2 == 1) || (x._1 < 0 && x._2 == 0))
        true
      else
        false
    }.count().toDouble / svmScoreAndLabels.count().toDouble
    println("SVM Test Error = " + (1.0 - svmAcc))

    /* FINISH RUNNING SVM */
    sc.stop()
  }
}
