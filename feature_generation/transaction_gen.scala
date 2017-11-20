import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import scala.collection.mutable.ListBuffer


object transaction_gen {
  // DateTime transformer
  val format = new java.text.SimpleDateFormat("yyyyMMdd")
  def get_days(a: String, b:String): Double = {
    val t1 = format.parse (a)
    val t2 = format.parse (b)
    (t1.getTime - t2.getTime) / (1000 * 60 * 60 * 24)
  }

  def get_svd(list: ListBuffer[Double], mean: Double): Double ={
    list.map( x => (x-mean)*(x-mean))
    Math.sqrt(list.sum / list.length)
  }
  // val test = List((1231,0), (1220,0), (1201,1), (1129, 0), (1115, 1), (1102,0), (1005,1))

  // return (index of (is_renew == 1 && is_cancel == 0))
  def renew_index(index: Int, list: List[(String, String, String, String, String)]): (Int) = {
    if (index > list.length - 1) index
    else if (list(index)._2 == "1" && list(index)._5 == "0") index
    else renew_index(index + 1, list)
  }

  def suspension_index(index: Int, list: List[(String, String, String, String, String)]): (Int) = {
    if (index > list.length - 1) (index)
    else if (list(index)._2 == "0" || list(index)._5 == "1") (index)
    else suspension_index(index + 1, list)
  }

  // return (renew, renew_avg, renew_svd)
  def get_renews(list: List[(String, String, String, String, String)]): List[Double]={
    var index = 1
    var renew_list_buffer = new ListBuffer[Double]()
    var cur = list.head
    while( index < list.length){
      val next_index = renew_index(index, list)
      if (next_index < list.length){
        renew_list_buffer += get_days(cur._3, list(next_index)._3)
        cur = list(next_index)
        index = next_index + 1
      }else{
        index = next_index + 1
      }
    }

    if (renew_list_buffer.isEmpty){
      renew_list_buffer += 0
    }
    val renew_mean = renew_list_buffer.sum / renew_list_buffer.length
    val renew_svd = get_svd(renew_list_buffer, renew_mean)
    List(renew_list_buffer.head, renew_mean, renew_svd)
  }

  def get_suspensions(list: List[(String, String, String, String, String)]): List[Double]={
    var index = 1
    var suspension_list_buffer = new ListBuffer[Double]()
    var cur = list.head
    while( index < list.length){
      val next_index = suspension_index(index, list)
      if (next_index < list.length){
        suspension_list_buffer += get_days(cur._3, list(next_index)._3)
        cur = list(next_index)
        index = next_index + 1
      }else{
        index = next_index + 1
      }
    }

    if (suspension_list_buffer.isEmpty){
      suspension_list_buffer += 0
    }

    val suspension_mean = suspension_list_buffer.sum / suspension_list_buffer.length
    val suspension_svd = get_svd(suspension_list_buffer, suspension_mean)
    List(suspension_list_buffer.head, suspension_mean, suspension_svd)
  }

  def get_expires(list: List[(String, String, String, String, String)]): List[Double]={
    var index = 1
    var expire_list_buffer = new ListBuffer[Double]()
    var cur = list.head
    while( index < list.length){
      val next_index = renew_index(index, list)
      // find previous expiration date > current transaction date
      if (get_days(list(next_index)._4, cur._3) > 0) {
        if (next_index < list.length) {
          expire_list_buffer += get_days(list(next_index)._4, cur._3)
          cur = list(next_index)
          index = next_index + 1
        } else {
          index = next_index + 1
        }
      }else{
        index = next_index + 1
      }
    }

    if( expire_list_buffer.isEmpty){
      expire_list_buffer += 0
    }

    val expire_mean = expire_list_buffer.sum / expire_list_buffer.length
    val expire_svd = get_svd(expire_list_buffer, expire_mean)
    List(expire_list_buffer.head, expire_mean, expire_svd)
  }

  // return (renew, renew_avg, renew_svd, suspension, suspension_avg, suspension_svd, expire, expire_avg, expire_svd)
  def get_tuple(list: List[(String, String, String, String, String)]): List[Any] ={
    get_renews(list):::get_suspensions(list):::get_expires(list)
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val data = sc.textFile("/Users/jimmy/Desktop/PA/raw_data/transactions.csv")
    val header = data.first()

    // tran: (msno, List[(msno, is_auto_renew, transaction_data, membership_expire_date, is_cancel)]
    val tran = data.filter(_!= header).map(_.split(",")).map(line => (line(0), line(5), line(6), line(7), line(8))).keyBy(_._1).groupByKey()

    // each key sort transaction_data desc
    val tran1 = tran.mapValues(iter => iter.toList.sortWith(_._3 > _._3))

    val tran2 = tran1.map( x => (x._1, get_tuple(x._2)))

    tran2.take(10).foreach(println(_))
  }
}

