# K8s data set

1. **Initial Investigation:**
   ```bash
   kubectl get pods -o wide
   kubectl describe pod <pod-name>
   kubectl logs <pod-name> --previous
   ```

2. **Check Common Issues:**
   - Resource constraints (CPU/memory limits)
   - Missing environment variables or ConfigMaps
   - Failed health/readiness probes
   - Image pull failures or wrong tags
   - Application startup failures

3. **Deep Dive Steps:**
   - Verify resource requests vs node capacity
   - Check probe configurations (initialDelaySeconds, timeouts)
   - Review recent deployments and changes
   - Test connectivity to dependencies
   - Examine application-specific logs

4. **Resolution Actions:**
   - Adjust resource limits if needed
   - Fix configuration issues
   - Update probe settings
   - Rollback problematic deployments
   - Scale nodes if capacity constrained"""
            },
            {
                "instruction": "Load balancer is returning 503 errors. How do I diagnose this?",
                "response": """To diagnose load balancer 503 errors, follow this systematic approach:

1. **Check Backend Health:**
   ```bash
   kubectl get endpoints <service-name>
   kubectl describe service <service-name>
   ```

2. **Verify Service Configuration:**
   - Ensure service selector matches pod labels
   - Check port configurations and target ports
   - Test direct pod connectivity with port-forward

3. **Health Check Analysis:**
   - Review readiness probe configurations
   - Test probe endpoints manually
   - Check probe timing (initialDelay, period, timeout)
   - Monitor health check logs

4. **Capacity Issues:**
   - Check if pods are overwhelmed
   - Review HPA status and scaling
   - Monitor resource utilization
   - Check connection pool exhaustion

5. **Network Investigation:**
   - Test DNS resolution within cluster
   - Check ingress controller logs
   - Verify security groups/firewall rules
   - Review load balancer access logs"""
            },
            {
                "instruction": "Application is consuming too much memory and getting OOMKilled. What's the investigation process?",
                "response": """Memory investigation process for OOMKilled containers:

1. **Immediate Analysis:**
   ```bash
   kubectl top pods --containers
   kubectl describe pod <pod-name>
   kubectl get events --field-selector involvedObject.name=<pod-name>
   ```

2. **Resource Configuration Review:**
   - Check memory requests vs limits
   - Compare actual usage to limits
   - Review QoS class (Guaranteed/Burstable/BestEffort)
   - Analyze historical memory patterns

3. **Application-Level Investigation:**
   - Generate memory dumps (heap dumps for Java)
   - Use profiling tools specific to your language
   - Check for memory leaks in application logs
   - Review garbage collection patterns

4. **System-Level Checks:**
   - Node memory pressure conditions
   - Available memory on nodes
   - Swap usage and thrashing
   - Other processes competing for memory

5. **Optimization Actions:**
   - Right-size memory requests and limits
   - Implement memory monitoring
   - Optimize application memory usage
   - Consider horizontal scaling
   - Review garbage collection tuning"""
            },
            {
                "instruction": "How do I implement effective circuit breakers in a microservices architecture?",
                "response": """Implementing circuit breakers for microservices resilience:

1. **Choose Circuit Breaker Library:**
   - Java: Resilience4j, Hystrix
   - Python: pybreaker, tenacity
   - Node.js: opossum
   - Go: sony/gobreaker

2. **Configuration Strategy:**
   ```yaml
   circuit_breaker:
     failure_rate_threshold: 50%
     slow_call_rate_threshold: 50%
     slow_call_duration_threshold: 60s
     minimum_number_of_calls: 10
     sliding_window_size: 10
     wait_duration_in_open_state: 60s
   ```

3. **Implementation Patterns:**
   - Wrap external service calls
   - Implement different strategies per service
   - Configure fallback mechanisms
   - Add monitoring and metrics
   - Set up alerting for state changes

4. **Fallback Strategies:**
   - Return cached responses
   - Provide default values
   - Degrade functionality gracefully
   - Queue requests for later processing
   - Use alternative service endpoints

5. **Monitoring and Observability:**
   - Track circuit breaker state transitions
   - Monitor failure rates and response times
   - Dashboard for circuit breaker health
   - Alert on prolonged open states
   - Analyze patterns for capacity planning"""