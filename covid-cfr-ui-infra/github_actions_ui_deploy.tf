#############################################
# GitHub Actions -> S3 (UI) + CloudFront
# OIDC (no long-lived AWS keys)
#############################################

variable "github_owner" {
  type    = string
  default = "achandrasek6"
}

variable "github_repo" {
  type    = string
  default = "Covid-Mortality-Prediction"
}

# Lock role assumption to a specific branch (recommended)
variable "github_branch" {
  type    = string
  default = "main"
}

# Your existing outputs / known values:
variable "ui_bucket_name" {
  type    = string
  default = "covid-cfr-ui-dev"
}

variable "ui_cloudfront_distribution_id" {
  type    = string
  default = "E32AEFURMCRZO"
}

data "aws_caller_identity" "current" {}

locals {
  github_repo_full = "${var.github_owner}/${var.github_repo}"

  # Only allow role assumption from this repo + branch
  github_sub = "repo:${local.github_repo_full}:ref:refs/heads/${var.github_branch}"

  cloudfront_distribution_arn = "arn:aws:cloudfront::${data.aws_caller_identity.current.account_id}:distribution/${var.ui_cloudfront_distribution_id}"
}

# --- 1) OIDC Provider for GitHub Actions ---
resource "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"

  client_id_list = [
    "sts.amazonaws.com"
  ]

  # This is commonly the GitHub Actions OIDC root CA thumbprint.
  # If AWS rejects it, see note below on how to verify the current thumbprint.
  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1"
  ]
}

# --- 2) IAM Role assumed by GitHub Actions ---
resource "aws_iam_role" "github_actions_ui_deploy" {
  name = "github-actions-ui-deploy-${var.project_name}-${var.env}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "GitHubActionsAssumeRoleWithOIDC"
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.github.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            "token.actions.githubusercontent.com:sub" = local.github_sub
          }
        }
      }
    ]
  })
}

# --- 3) Policy: allow S3 sync into the UI bucket + CloudFront invalidation ---
resource "aws_iam_policy" "github_actions_ui_deploy_policy" {
  name        = "github-actions-ui-deploy-policy-${var.project_name}-${var.env}"
  description = "Allows GitHub Actions to deploy UI to S3 and invalidate CloudFront."

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # S3: list bucket + upload/delete objects
      {
        Sid    = "S3ListBucket"
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = "arn:aws:s3:::${var.ui_bucket_name}"
      },
      {
        Sid    = "S3ObjectRW"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:PutObjectTagging",
          "s3:DeleteObject",
          "s3:GetObject",
          "s3:GetObjectTagging"
        ]
        Resource = "arn:aws:s3:::${var.ui_bucket_name}/*"
      },

      # CloudFront: invalidate
      {
        Sid    = "CloudFrontInvalidate"
        Effect = "Allow"
        Action = [
          "cloudfront:CreateInvalidation"
        ]
        Resource = local.cloudfront_distribution_arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_ui_deploy" {
  role       = aws_iam_role.github_actions_ui_deploy.name
  policy_arn = aws_iam_policy.github_actions_ui_deploy_policy.arn
}

output "github_actions_ui_deploy_role_arn" {
  value = aws_iam_role.github_actions_ui_deploy.arn
}
