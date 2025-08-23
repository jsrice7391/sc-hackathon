#!/bin/bash
set -e

# Variables - update these as needed
AWS_REGION="us-east-1"
ECR_REPO="what-we-play-for"
IMAGE_TAG="latest"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Build Docker image
docker build -t $ECR_REPO:$IMAGE_TAG .

# Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Create ECR repo if it doesn't exist
aws ecr describe-repositories --repository-names $ECR_REPO --region $AWS_REGION > /dev/null 2>&1 || \
    aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION

# Tag and push image
docker tag $ECR_REPO:$IMAGE_TAG $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG

echo "Docker image pushed to ECR: $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG"