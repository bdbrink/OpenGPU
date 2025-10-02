# Kubernetes pods stuck in CrashLoopBackOff. Walk me through troubleshooting.

Here's a systematic approach:

1. **Initial Investigation:**
   ```bash
   kubectl get pods -o wide
   kubectl describe pod <pod-name>
   kubectl logs <pod-name> --previous
   ```

2. **Common Issues:**
   - Resource constraints
   - Missing environment variables
   - Failed health probes

---

# Load balancer returning 503 errors. How to diagnose?

Diagnostic steps:

1. **Check Backend Health:**
   ```bash
   kubectl get endpoints <service-name>
   ```

2. **Verify Configuration:**
   - Service selector matches pod labels
   - Port configurations are correct
