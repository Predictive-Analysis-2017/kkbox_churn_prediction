#!/bin/sh
spark-submit --driver-memory 12G --executor-memory 12G --executor-cores 10 --driver-cores 10 --class "pa_final" /home/hc2264/final/kkbox_churn_prediction/src/pa_final/target/scala-2.10/pa-final_2.10-1.0.jar > clean.log
