package controllers

import (
	"context"
	"github.com/gin-gonic/gin"
	"github.com/minio/minio-go/v7"
	"github.com/rs/zerolog/log"
	"net/http"
)

func Upload(minioClient *minio.Client) gin.HandlerFunc {
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

		err = minioClient.MakeBucket(ctx, bucketName, minio.MakeBucketOptions{})
		if err != nil {
			exists, errBucketExists := minioClient.BucketExists(ctx, bucketName)
			if errBucketExists == nil && exists {
				log.Info().Str("bucketName", bucketName).Msg("Bucket exists.")
			} else {
				log.Error().Err(err).Msg("minioClient.MakeBucket")
				c.JSON(http.StatusInternalServerError, gin.H{
					"error": err.Error(),
				})
				return
			}
		} else {
			log.Info().Str("bucketName", bucketName).Msg("Bucket created.")
		}

		uploadInfo, err := minioClient.PutObject(ctx, bucketName, fileHeader.Filename, file, fileHeader.Size, minio.PutObjectOptions{})
		if err != nil {
			log.Error().Err(err).Msg("minioClient.FPutObject")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": err.Error(),
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"bucket":           uploadInfo.Bucket,
			"key":              uploadInfo.Key,
			"etag":             uploadInfo.ETag,
			"size":             uploadInfo.Size,
			"lastModified":     uploadInfo.LastModified,
			"location":         uploadInfo.Location,
			"versionID":        uploadInfo.VersionID,
			"expiration":       uploadInfo.Expiration,
			"expirationRuleID": uploadInfo.ExpirationRuleID,
		})
	}
	return fn
}
