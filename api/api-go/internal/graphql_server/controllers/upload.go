package controllers

import (
	"context"
	"errors"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
	"net/http"
)

func Upload(s3Client *s3.Client) gin.HandlerFunc {
	fn := func(c *gin.Context) {
		fileHeader, err := c.FormFile("file")
		if err != nil {
			log.Error().Err(err).Msg("c.FormFile")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": err.Error(),
			})
			return
		}
		file, err := fileHeader.Open()
		if err != nil {
			log.Error().Err(err).Msg("file.Open")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": err.Error(),
			})
			return
		}

		ctx := context.Background()
		bucketName := "production-hm-bucket"

		_, err = s3Client.CreateBucket(ctx, &s3.CreateBucketInput{
			Bucket: aws.String(bucketName),
		})
		if err != nil {
			var bucketAlreadyExists *types.BucketAlreadyExists
			var bucketAlreadyOwnedByYou *types.BucketAlreadyOwnedByYou
			if errors.As(err, &bucketAlreadyExists) || errors.As(err, &bucketAlreadyOwnedByYou) {
				log.Info().Str("bucketName", bucketName).Msg("Bucket exists.")
			} else {
				log.Error().Err(err).Msg("s3Client.CreateBucket")
				c.JSON(http.StatusInternalServerError, gin.H{
					"error": err.Error(),
				})
				return
			}
		} else {
			log.Info().Str("bucketName", bucketName).Msg("Bucket created.")
		}

		putObjectOutput, err := s3Client.PutObject(ctx, &s3.PutObjectInput{
			Bucket:        aws.String(bucketName),
			Key:           aws.String(fileHeader.Filename),
			Body:          file,
			ContentLength: aws.Int64(fileHeader.Size),
		})
		if err != nil {
			log.Error().Err(err).Msg("s3Client.PutObject")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": err.Error(),
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"bucket":    bucketName,
			"key":       fileHeader.Filename,
			"etag":      aws.ToString(putObjectOutput.ETag),
			"versionID": aws.ToString(putObjectOutput.VersionId),
		})
	}
	return fn
}
