terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-2"
}

# Just a name to keep things organized
variable "project_name" {
  type    = string
  default = "covid-cfr"
}

variable "env" {
  type    = string
  default = "dev"
}

# --- S3 bucket for UI (private; served via CloudFront + OAC) ---

resource "aws_s3_bucket" "ui_bucket" {
  bucket = "${var.project_name}-ui-${var.env}"
}

resource "aws_s3_bucket_versioning" "ui_versioning" {
  bucket = aws_s3_bucket.ui_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ui_sse" {
  bucket = aws_s3_bucket.ui_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Strongly prevent accidental public exposure
resource "aws_s3_bucket_public_access_block" "ui_block_public" {
  bucket = aws_s3_bucket.ui_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

output "ui_bucket_name" {
  value = aws_s3_bucket.ui_bucket.bucket
}

# --- CloudFront + Origin Access Control (OAC) ---

resource "aws_cloudfront_origin_access_control" "ui_oac" {
  name                              = "${var.project_name}-ui-oac-${var.env}"
  description                       = "OAC for ${var.project_name} UI bucket"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# AWS managed cache policy: "Managed-CachingOptimized"
data "aws_cloudfront_cache_policy" "caching_optimized" {
  name = "Managed-CachingOptimized"
}

resource "aws_cloudfront_distribution" "ui_cdn" {
  enabled             = true
  comment             = "${var.project_name} UI (${var.env})"
  default_root_object = "index.html"
  price_class         = "PriceClass_100"

  origin {
    domain_name              = aws_s3_bucket.ui_bucket.bucket_regional_domain_name
    origin_id                = "s3-ui-origin"
    origin_access_control_id = aws_cloudfront_origin_access_control.ui_oac.id
  }

  default_cache_behavior {
    target_origin_id       = "s3-ui-origin"
    viewer_protocol_policy = "redirect-to-https"
    compress               = true

    allowed_methods = ["GET", "HEAD", "OPTIONS"]
    cached_methods  = ["GET", "HEAD"]

    cache_policy_id = data.aws_cloudfront_cache_policy.caching_optimized.id
  }

  # SPA routing: return index.html on 403/404 so refresh/deep-links work
  custom_error_response {
    error_code            = 403
    response_code         = 200
    response_page_path    = "/index.html"
    error_caching_min_ttl = 0
  }

  custom_error_response {
    error_code            = 404
    response_code         = 200
    response_page_path    = "/index.html"
    error_caching_min_ttl = 0
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  # Use the default CloudFront certificate (HTTPS) for now.
  # If/when you add a custom domain, swap this to an ACM cert in us-east-1.
  viewer_certificate {
    cloudfront_default_certificate = true
  }

  depends_on = [aws_s3_bucket_public_access_block.ui_block_public]
}

# Allow ONLY this CloudFront distribution to read objects from the bucket
resource "aws_s3_bucket_policy" "ui_allow_cloudfront_read" {
  bucket = aws_s3_bucket.ui_bucket.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowCloudFrontReadOnly"
        Effect = "Allow"
        Principal = {
          Service = "cloudfront.amazonaws.com"
        }
        Action   = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.ui_bucket.arn}/*"
        Condition = {
          StringEquals = {
            "AWS:SourceArn" = aws_cloudfront_distribution.ui_cdn.arn
          }
        }
      }
    ]
  })

  depends_on = [aws_cloudfront_distribution.ui_cdn]
}

output "ui_cloudfront_domain" {
  value = aws_cloudfront_distribution.ui_cdn.domain_name
}

output "ui_cloudfront_distribution_id" {
  value = aws_cloudfront_distribution.ui_cdn.id
}
