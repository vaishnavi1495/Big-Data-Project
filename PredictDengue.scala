import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.types.{DoubleType, StringType, StructField}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.regression.LinearRegression
import scala.collection.mutable.{ArrayBuffer, ListMap}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD

object PredictDengue {
  def main(args: Array[String]):Unit = {
    if (args.length != 3) {
      println("I need two parameters ")
    }
    //Initializing spark env variables
    val sc = new SparkContext(new SparkConf().setAppName("DengAI"))
    val spark = SparkSession.builder().getOrCreate()
    val sqlContext = SparkSession.builder().getOrCreate()

    //Reading the features
    val df1 = spark.read.option("header", true).csv(args(0))

    //Reading labels
    val df2 = spark.read.option("header", true).csv(args(1))

    //Joining features and labels without column redundancy - Using seq
    val df4 = df1.join(df2, Seq("city", "year", "weekofyear"), "inner")

    //Casting the columns as double because ML algorithms need double datatype to operate

    val df3 = df4.select(df4("city").cast(StringType).as("city"), df4("ndvi_ne").cast(DoubleType).as("ndvi_ne"), df4("ndvi_nw").cast(DoubleType).as("ndvi_nw"),                      df4("ndvi_se").cast(DoubleType).as("ndvi_se"),
      df4("ndvi_sw").cast(DoubleType).as("ndvi_sw"),
      df4("precipitation_amt_mm").cast(DoubleType).as("precipitation_amt_mm"),
      df4("reanalysis_air_temp_k").cast(DoubleType).as("reanalysis_air_temp_k"),
      df4("reanalysis_avg_temp_k").cast(DoubleType).as("reanalysis_avg_temp_k"),
      df4("reanalysis_dew_point_temp_k").cast(DoubleType).as("reanalysis_dew_point_temp_k"),
      df4("reanalysis_max_air_temp_k").cast(DoubleType).as("reanalysis_max_air_temp_k"),
      df4("reanalysis_min_air_temp_k").cast(DoubleType).as("reanalysis_min_air_temp_k"),
      df4("reanalysis_precip_amt_kg_per_m2").cast(DoubleType).as("reanalysis_precip_amt_kg_per_m2"),
      df4("reanalysis_relative_humidity_percent").cast(DoubleType).as("reanalysis_relative_humidity_percent"),
      df4("reanalysis_sat_precip_amt_mm").cast(DoubleType).as("reanalysis_sat_precip_amt_mm"),
      df4("reanalysis_specific_humidity_g_per_kg").cast(DoubleType).as("reanalysis_specific_humidity_g_per_kg"),
      df4("reanalysis_tdtr_k").cast(DoubleType).as("reanalysis_tdtr_k"),
      df4("station_avg_temp_c").cast(DoubleType).as("station_avg_temp_c"),
      df4("station_diur_temp_rng_c").cast(DoubleType).as("station_diur_temp_rng_c"),
      df4("station_max_temp_c").cast(DoubleType).as("station_max_temp_c"),
      df4("station_min_temp_c").cast(DoubleType).as("station_min_temp_c"),
      df4("station_precip_mm").cast(DoubleType).as("station_precip_mm"),
      df4("total_cases").cast(DoubleType).as("total_cases"))

    //Since our data may contain null values, we will use the maximum occuring value of that feature to fill it.
    //Here we are calculating maximum occuring value in each feature.
    import org.apache.spark.sql.functions._
    import scala.collection.mutable.ArrayBuffer
    val grouping = Seq("station_min_temp_c", "total_cases", "reanalysis_tdtr_k", "station_max_temp_c", "ndvi_nw", "reanalysis_air_temp_k", "reanalysis_min_air_temp_k", "reanalysis_precip_amt_kg_per_m2", "precipitation_amt_mm", "reanalysis_dew_point_temp_k", "station_precip_mm", "ndvi_ne", "reanalysis_max_air_temp_k", "ndvi_sw", "ndvi_se", "reanalysis_specific_humidity_g_per_kg", "reanalysis_avg_temp_k", "station_diur_temp_rng_c", "reanalysis_relative_humidity_percent", "station_avg_temp_c", "reanalysis_sat_precip_amt_mm")
    var topValues = ArrayBuffer[Double]()
    for(colIndex <- 0 to 20)
    {
      val temp = df3.select(grouping(colIndex)).sort(desc(grouping(colIndex))).limit(1)
      topValues += temp.first().getDouble(0)
    }

    //Fill in the null values - we are doing this because we will have to find corealtion between features and labels later.
    val df_without_null = df3
      .na.fill(topValues(0), Seq("station_min_temp_c"))
      .na.fill(topValues(1), Seq("total_cases"))
      .na.fill(topValues(2), Seq("reanalysis_tdtr_k"))
      .na.fill(topValues(3), Seq("station_max_temp_c"))
      .na.fill(topValues(4), Seq("ndvi_nw"))
      .na.fill(topValues(5), Seq("reanalysis_air_temp_k"))
      .na.fill(topValues(6), Seq("reanalysis_min_air_temp_k"))
      .na.fill(topValues(7), Seq("reanalysis_precip_amt_kg_per_m2"))
      .na.fill(topValues(8), Seq("precipitation_amt_mm"))
      .na.fill(topValues(9), Seq("reanalysis_dew_point_temp_k"))
      .na.fill(topValues(10), Seq("station_precip_mm"))
      .na.fill(topValues(11), Seq("ndvi_ne"))
      .na.fill(topValues(12), Seq("reanalysis_max_air_temp_k"))
      .na.fill(topValues(13), Seq("ndvi_sw"))
      .na.fill(topValues(14), Seq("ndvi_se"))
      .na.fill(topValues(15), Seq("reanalysis_specific_humidity_g_per_kg"))
      .na.fill(topValues(16), Seq("reanalysis_avg_temp_k"))
      .na.fill(topValues(17), Seq("station_diur_temp_rng_c"))
      .na.fill(topValues(18), Seq("reanalysis_relative_humidity_percent"))
      .na.fill(topValues(19), Seq("station_avg_temp_c"))
      .na.fill(topValues(20), Seq("reanalysis_sat_precip_amt_mm"))

    //Since the models required for two cities may be different, we will create separate dataframes for the two cities.
    val df_sj = df_without_null.filter(df_without_null("city")==="sj") //Filtering for san juan
    val df_iq = df_without_null.filter(df_without_null("city")==="iq") //filtering for iquitos

    //Finding corelation between features and label
    val corr_min = df_sj.stat.corr("total_cases", "reanalysis_tdtr_k")
    val corr_max_1 = df_sj.stat.corr("total_cases", "reanalysis_specific_humidity_g_per_kg")
    val corr_max_2 = df_sj.stat.corr("total_cases", "reanalysis_dew_point_temp_k")
    val corr_max_3 = df_sj.stat.corr("total_cases", "reanalysis_min_air_temp_k")
    val corr_max_4 = df_sj.stat.corr("total_cases", "station_min_temp_c")

    //println("The features which strongly corelate with the label are: ")
    //println("reanalysis_specific_humidity_g_per_kg : " + corr_max_1)
    //println("reanalysis_dew_point_temp_k: "+ corr_max_2)
    //println("reanalysis_min_air_temp_k: "+ corr_max_3)
    //println("station_min_temp_c: " + corr_max_4)

    //We will select those four features as they stronly corealte with the label.
    val finaldf_sj = df_sj.select("reanalysis_specific_humidity_g_per_kg", "reanalysis_dew_point_temp_k", "reanalysis_min_air_temp_k", "station_min_temp_c", "total_cases")
    val finaldf_iq = df_iq.select("reanalysis_specific_humidity_g_per_kg", "reanalysis_dew_point_temp_k", "reanalysis_min_air_temp_k", "station_min_temp_c", "total_cases")

    //Since the data is in csv format and for regression we need to convert it into libSVM format we will do some manipulations
    //We will create a RDD

    //only doubles accepted by sparse vector, so that's what we filter for
    val fieldSeq: scala.collection.Seq[StructField] = finaldf_sj.schema.fields.toSeq.filter(f => f.dataType == DoubleType)
    val fieldNameSeq: Seq[String] = fieldSeq.map(f => f.name)

    var positionsArray: ArrayBuffer[LabeledPoint] = ArrayBuffer[LabeledPoint]()

    finaldf_sj.collect().foreach
    {

      //We are converting each row to a labeledPoint format.
      row => positionsArray+=convertRowToLabeledPoint(row,fieldNameSeq,row.getAs("total_cases"));

    }

    //We create an RDD out of those labeled points.
    val mRdd:RDD[LabeledPoint]= spark.sparkContext.parallelize(positionsArray.toSeq)

    //Converting the rdd to dataframe.
    import sqlContext.implicits._
    val myFinalDf = mRdd.toDF()

    //Split the data frame into test and training set
    val Array(train_sj_new, test_sj_new) = myFinalDf.randomSplit(Array(0.8,0.2))

    //Create a logistic regression model
    val lr = new LinearRegression()
      .setMaxIter(1000)
      .setRegParam(0.1)
      .setElasticNetParam(0.8)

    //Train the training set
    val lr_model = lr.fit(train_sj_new)

    //Transform the testing set
    val pred_test = lr_model.transform(test_sj_new)

    //Declaring a regression evaluator with mae as the metric name
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("mae")

    //Evaluate the error
    val error = evaluator.evaluate(pred_test)

    var output = ""
    output+="Error of the model for san juan is :"+ error+"\n"

    //We will do the above steps once again for iquitos
    //only doubles accepted by sparse vector, so that's what we filter for
    val fieldSeq_iq: scala.collection.Seq[StructField] = finaldf_iq.schema.fields.toSeq.filter(f => f.dataType == DoubleType)
    val fieldNameSeq_iq: Seq[String] = fieldSeq_iq.map(f => f.name)

    var positionsArray_iq: ArrayBuffer[LabeledPoint] = ArrayBuffer[LabeledPoint]()

    finaldf_iq.collect().foreach
    {

      //We are converting each row to a labeledPoint format.
      row => positionsArray_iq+=convertRowToLabeledPoint(row,fieldNameSeq_iq,row.getAs("total_cases"));

    }

    //We create an RDD out of those labeled points.
    val mRdd_iq:RDD[LabeledPoint]= spark.sparkContext.parallelize(positionsArray_iq.toSeq)

    //Converting the rdd to dataframe.
    import sqlContext.implicits._
    val myFinalDf_iq = mRdd_iq.toDF()

    //Split the data frame into test and training set
    val Array(train_iq_new, test_iq_new) = myFinalDf_iq.randomSplit(Array(0.8,0.2))

    //Train the training set
    val lr_model_iq = lr.fit(train_iq_new)

    //Transform the testing set
    val pred_test_iq = lr_model_iq.transform(test_iq_new)

    //Evaluate the error
    val error_iq = evaluator.evaluate(pred_test_iq)

    output+="Error of the model for iquitos is :"+ error_iq+"\n"

    sc.parallelize(List(output)).saveAsTextFile(args(2))
  }

  //This function takes in a row, column names and label, it returns a labledpoint which is suitable for svm format
  def convertRowToLabeledPoint(rowIn: Row, fieldNameSeq: Seq[String], label:Double): LabeledPoint =
  {
    try
    {
      //Get the values of each column.
      val values: Map[String, Double] = rowIn.getValuesMap(fieldNameSeq)

      val sortedValuesMap = ListMap(values.toSeq.sortBy(_._1): _*)
      val rowValuesItr: Iterable[Double] = sortedValuesMap.values

      var positionsArray: ArrayBuffer[Int] = ArrayBuffer[Int]()
      var valuesArray: ArrayBuffer[Double] = ArrayBuffer[Double]()
      var currentPosition: Int = 0
      //Here we are storing what value is at what position.
      rowValuesItr.foreach
      {
        kv =>
          if (kv >=0)
          {
            valuesArray += kv;
            positionsArray += currentPosition;
          }
          //println("The kv for " + currentPosition + "is " +kv)
          currentPosition = currentPosition + 1;
      }
      //Creating a sparse vectors to store values at particular position only.
      val lp:LabeledPoint = new LabeledPoint(label,Vectors.sparse(positionsArray.size,positionsArray.toArray, valuesArray.toArray))
      //Returnnig the labledPoint, a format suitable for regression which uses SVM.
      return lp

    }
    catch
      {
        case ex: Exception =>
        {
          throw new Exception(ex)
        }
      }
  }

}
