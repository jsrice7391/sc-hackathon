variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "fastapi-ollama-cluster"
}

variable "vpc_id" {
  description = "VPC ID for EKS"
  type        = string
}

variable "subnets" {
  description = "Subnets for EKS nodes"
  type        = list(string)
}
