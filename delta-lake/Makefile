delta-lake-spark-scala:
	spark-shell --packages io.delta:delta-core_2.12:2.3.0 \
		--conf=spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
		--conf=spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog

delta-lake-spark-sql:
	kubectl exec --stdin --tty --namespace=hm-spark spark-master-0 -- \
		spark-sql --packages io.delta:delta-core_2.12:2.3.0 \
			--conf=spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
			--conf=spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog
delta-lake-spark-sql-s3:
	kubectl exec --stdin --tty --namespace=hm-spark spark-master-0 -- \
		spark-sql --packages io.delta:delta-core_2.12:2.3.0 \
			--conf=spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
			--conf=spark.hadoop.fs.s3a.access.key=xxx \
			--conf=spark.hadoop.fs.s3a.secret.key=xxx \
			--conf=spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog
