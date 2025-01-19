amazon-emr-cluster-list:
	aws emr list-clusters
amazon-emr-cluster-describe:
	aws emr describe-cluster --cluster-id=xxx
amazon-emr-cluster-terminate:
	aws emr terminate-clusters --cluster-ids=xxx

amazon-emr-studio-list:
	aws emr list-studios
amazon-emr-studio-describe:
	aws emr describe-studio --studio-id=xxx

# Amazon EMR - Hadoop
hadoop-yarn-status:
	sudo systemctl status hadoop-yarn-resourcemanager
hadoop-yarn-start:
	sudo systemctl start hadoop-yarn-resourcemanager
hadoop-yarn-stop:
	sudo systemctl stop hadoop-yarn-resourcemanager

# Amazon EMR - Hadoop HDFS
hadoop-hdfs-version:
	hdfs version
hadoop-hdfs-ls:
	hdfs dfs -ls
	hdfs dfs -ls /
	hdfs dfs -ls /user
	hdfs dfs -ls hdfs:///
	hdfs dfs -ls hdfs:///user
	hdfs dfs -ls s3://hm-production-bucket/
hadoop-hdfs-show-disk-free:
	hdfs dfs -df
hadoop-hdfs-show-disk-usage:
	hdfs dfs -du
hadoop-hdfs-copy-from-local:
	echo "Hello" >> file.txt
	hdfs dfs -copyFromLocal file.txt
hadoop-hdfs-cat:
	hdfs dfs -cat file.txt
hadoop-hdfs-mv:
	hdfs dfs -mv file.txt file2.txt
hadoop-hdfs-cp:
	hdfs dfs -cp file.txt file2.txt
hadoop-hdfs-rm:
	hdfs dfs -rm file
	hdfs dfs -rm -r folder
