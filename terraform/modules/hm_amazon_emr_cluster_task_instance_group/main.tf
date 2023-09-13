# https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/emr_instance_group
resource "aws_emr_instance_group" "hm_amazon_emr_cluster_task_instance_group" {
  name           = "Task"
  cluster_id     = var.amazon_emr_cluster_id
  instance_type  = var.task_instance_type
  instance_count = var.task_instance_count
  bid_price      = 1.0
}
