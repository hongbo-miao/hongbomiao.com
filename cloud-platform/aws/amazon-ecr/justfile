# https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html

amazon-ecr-login:
	aws ecr get-login-password --region=us-west-2 | \
	docker login \
		--username=AWS \
		--password-stdin \
		272394222652.dkr.ecr.us-west-2.amazonaws.com
amazon-ecr-push:
	# docker pull docker.io/alpine:latest
	# docker tag docker.io/alpine:latest 272394222652.dkr.ecr.us-west-2.amazonaws.com/alpine:latest
	docker push 272394222652.dkr.ecr.us-west-2.amazonaws.com/alpine:latest
