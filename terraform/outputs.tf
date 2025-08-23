output "kubeconfig" {
  value = module.eks.kubeconfig
  sensitive = true
}

