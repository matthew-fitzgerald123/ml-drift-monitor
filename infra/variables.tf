variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "ml-drift-monitor"
}

variable "db_instance_class" {
  type    = string
  default = "db.t3.micro"
}

variable "db_name" {
  type    = string
  default = "drift_monitor"
}

variable "db_username" {
  type    = string
  default = "drift_monitor"
}

variable "db_password" {
  type      = string
  sensitive = true
}

variable "p2_api_url" {
  description = "Base URL of the P2 ML platform feature store"
  type        = string
  default     = "http://localhost:8080"
}
