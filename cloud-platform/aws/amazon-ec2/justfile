amazon-ec2-spot-fleet-requests-describe:
	aws ec2 describe-spot-fleet-requests --spot-fleet-request-ids=sfr-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
amazon-ec2-spot-fleet-requests-cancel:
	aws ec2 cancel-spot-fleet-requests --spot-fleet-request-ids=sfr-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx --terminate-instances

amazon-ec2-storage-class-change:
	aws s3 cp s3://hm-production-bucket/data/raw/adsb_2x_flight_trace_data/ s3://hm-production-bucket/data/raw/adsb_2x_flight_trace_data/ --recursive --storage-class=STANDARD

amazon-vpc-list:
	aws ec2 describe-vpcs

amazon-security-group-list:
	aws ec2 describe-security-groups
