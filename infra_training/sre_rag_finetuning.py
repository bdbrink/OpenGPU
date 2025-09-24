#!/usr/bin/env python3
"""
SRE RAG & Fine-tuning Pipeline - Standalone Script
Works with your existing GPU detection and model loading setup
"""

import torch
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
import logging
from datetime import datetime
import pickle

# RAG-specific imports
try:
    import faiss
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    print("⚠️ Install RAG deps: pip install faiss-cpu sentence-transformers pandas")
    RAG_AVAILABLE = False

# Fine-tuning specific imports
try:
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from trl import SFTTrainer
    from transformers import TrainingArguments, DataCollatorForLanguageModeling
    FINETUNING_AVAILABLE = True
except ImportError:
    print("⚠️ Install fine-tuning deps: pip install datasets peft trl")
    FINETUNING_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    vector_db_path: str = "./rag_vectors"
    knowledge_base_path: str = "./knowledge_base"

@dataclass
class ModelInfo:
    """Container for model information passed from main script"""
    model: Any = None
    tokenizer: Any = None
    device: str = "cpu"
    gpu_info: Dict = None
    batch_size: int = 16
    use_mixed_precision: bool = False

class SREKnowledgeBase:
    """SRE Knowledge Management"""
    
    @staticmethod
    def get_sre_documents() -> List[Dict[str, str]]:
        """Comprehensive SRE knowledge base"""
        return [
            {
                "title": "Kubernetes Pod Troubleshooting Guide",
                "content": """
                Complete Kubernetes pod troubleshooting workflow:
                
                1. Initial Assessment:
                - kubectl get pods -o wide (check status, restarts, node)
                - kubectl describe pod <pod-name> (events, conditions)
                - kubectl logs <pod-name> --previous (last crash logs)
                
                2. Common CrashLoopBackOff Causes:
                - Insufficient resources (CPU/memory limits)
                - Missing or incorrect environment variables
                - Application startup failures
                - Failed health/readiness probes
                - Image pull failures or wrong image tags
                - Missing ConfigMaps or Secrets
                - Filesystem permission issues
                
                3. Systematic Debugging:
                - Check resource requests vs limits
                - Verify probe configurations (initialDelaySeconds, timeouts)
                - Validate service account permissions
                - Review network policies and connectivity
                - Check persistent volume claims and mounts
                
                4. Advanced Troubleshooting:
                - kubectl exec for interactive debugging
                - Port-forward for direct service testing
                - Review cluster events: kubectl get events
                - Check node conditions and capacity
                """
            },
            {
                "title": "Load Balancer and Ingress Troubleshooting",
                "content": """
                Load balancer 503/504 error investigation process:
                
                1. Service Health Verification:
                - Check backend service endpoints: kubectl get endpoints
                - Verify service selector matches pod labels
                - Test direct pod connectivity: kubectl port-forward
                - Review service type and port configurations
                
                2. Health Check Analysis:
                - Validate readiness probe configurations
                - Check probe endpoints return 200 OK
                - Verify probe timing (initialDelay, period, timeout)
                - Monitor health check logs in load balancer
                
                3. Capacity and Scaling Issues:
                - Check HPA (Horizontal Pod Autoscaler) status
                - Review resource utilization metrics
                - Verify cluster autoscaler functionality
                - Monitor connection pool exhaustion
                
                4. Network and Connectivity:
                - Test DNS resolution within cluster
                - Check service mesh configuration (Istio/Linkerd)
                - Verify security groups and firewall rules
                - Review ingress controller logs and configuration
                
                5. Load Balancer Specific:
                - Check target group health (AWS ALB/NLB)
                - Review load balancer access logs
                - Verify SSL certificate validity
                - Check rate limiting and WAF rules
                """
            },
            {
                "title": "Memory Management and OOM Investigation",
                "content": """
                Memory pressure and OOM (Out of Memory) troubleshooting:
                
                1. Container Memory Analysis:
                - kubectl top pods --containers (current usage)
                - kubectl describe pod <pod> (limits and requests)
                - Check OOMKilled events: kubectl get events
                - Review container restart patterns
                
                2. Application Memory Profiling:
                - Java: Generate heap dumps, analyze with MAT/VisualVM
                - Python: Use memory_profiler, tracemalloc
                - Node.js: --inspect flag with Chrome DevTools
                - Go: pprof for memory profiling
                
                3. System-Level Investigation:
                - Node memory usage: kubectl describe node
                - System memory: free -h, /proc/meminfo
                - Check for memory leaks in system processes
                - Review swap usage and thrashing
                
                4. Kubernetes Memory Management:
                - Understand requests vs limits impact
                - Review QoS classes (Guaranteed, Burstable, BestEffort)
                - Check node pressure conditions
                - Monitor memory overcommit ratios
                
                5. Optimization Strategies:
                - Right-size memory requests and limits
                - Implement memory monitoring and alerting
                - Use init containers for heavy initialization
                - Consider memory-efficient application patterns
                - Review garbage collection settings and tuning
                """
            },
            {
                "title": "Database Performance and Connection Issues",
                "content": """
                Database performance troubleshooting methodology:
                
                1. Connection Pool Management:
                - Monitor active vs idle connections
                - Check connection pool size configuration
                - Review connection timeout settings
                - Identify connection leaks in applications
                - Monitor connection establishment time
                
                2. Query Performance Analysis:
                - Enable slow query logging
                - Identify expensive queries with EXPLAIN plans
                - Review index usage and optimization
                - Check for missing indexes on frequently queried columns
                - Analyze query execution statistics
                
                3. Lock and Concurrency Issues:
                - Monitor deadlock frequency and causes
                - Check for long-running transactions
                - Review table locking patterns
                - Identify blocking queries and sessions
                - Analyze wait events and contention points
                
                4. Resource Utilization:
                - Monitor CPU usage patterns
                - Check memory allocation (buffer pools, caches)
                - Review disk I/O performance and latency
                - Monitor network throughput to database
                - Check for resource contention with other processes
                
                5. Replication and High Availability:
                - Monitor replication lag in read replicas
                - Check binlog/WAL shipping performance
                - Verify failover mechanisms and timeouts
                - Review backup and recovery procedures
                - Test disaster recovery scenarios regularly
                """
            },
            {
                "title": "Network Connectivity and DNS Troubleshooting",
                "content": """
                Network debugging toolkit for containerized environments:
                
                1. Basic Connectivity Testing:
                - ping for ICMP connectivity (if enabled)
                - telnet/nc for TCP port testing
                - curl/wget for HTTP endpoint testing
                - traceroute/mtr for path analysis
                
                2. DNS Resolution Debugging:
                - nslookup/dig for DNS queries
                - Check /etc/resolv.conf in containers
                - Test cluster DNS (usually kube-dns/CoreDNS)
                - Verify service discovery mechanisms
                - Review DNS caching and TTL issues
                
                3. Container Networking:
                - Check CNI plugin configuration
                - Review network policies and their effects
                - Monitor network interface statistics
                - Verify CIDR ranges and IP allocation
                - Test inter-node pod communication
                
                4. Service Mesh Troubleshooting:
                - Verify sidecar injection (Istio/Linkerd)
                - Check traffic routing rules
                - Review circuit breaker configurations
                - Monitor service-to-service authentication
                - Debug TLS/mTLS certificate issues
                
                5. Network Performance:
                - Use iperf3 for bandwidth testing
                - Monitor packet loss and latency
                - Check for network interface errors
                - Review QoS and traffic shaping
                - Analyze network security policies impact
                """
            },
            {
                "title": "Microservices Resilience Patterns",
                "content": """
                Building resilient microservices architectures:
                
                1. Circuit Breaker Pattern:
                - Implement with libraries like Hystrix, resilience4j
                - Configure failure thresholds and recovery timeouts
                - Monitor circuit breaker state transitions
                - Use different strategies for different failure types
                - Implement dashboard for circuit breaker status
                
                2. Retry and Backoff Strategies:
                - Exponential backoff with jitter
                - Circuit breaker integration with retries
                - Distinguish between retryable and non-retryable errors
                - Set maximum retry counts and timeouts
                - Implement retry budget to prevent cascading failures
                
                3. Bulkhead and Isolation:
                - Separate thread pools for different operations
                - Resource isolation between critical and non-critical paths
                - Use different connection pools for different services
                - Implement queue-based decoupling where appropriate
                - Isolate database connections by operation type
                
                4. Graceful Degradation:
                - Define fallback mechanisms for each service dependency
                - Implement feature flags for non-critical functionality
                - Cache responses for graceful degradation
                - Provide default values when external services fail
                - Design for partial functionality under load
                
                5. Observability and Monitoring:
                - Implement distributed tracing (Jaeger, Zipkin)
                - Use structured logging with correlation IDs
                - Monitor SLIs/SLOs for each service
                - Set up alerting for cascading failure detection
                - Dashboard for dependency health and performance
                """
            },
            {
                "title": "Incident Response and Troubleshooting Methodology",
                "content": """
                Systematic incident response approach for SRE teams:
                
                1. Incident Detection and Alerting:
                - Implement effective monitoring and alerting systems
                - Use SLI/SLO based alerting to reduce noise
                - Set up escalation policies and on-call rotations
                - Integrate with communication tools (Slack, PagerDuty)
                - Monitor for symptoms rather than just causes
                
                2. Initial Response (First 5 minutes):
                - Assess impact and severity quickly
                - Form incident command structure
                - Begin initial investigation and mitigation
                - Communicate status to stakeholders
                - Document timeline and decisions
                
                3. Investigation Methodology:
                - Follow the scientific method: hypothesis, test, conclude
                - Use distributed tracing to understand request flows
                - Correlate metrics across different system layers
                - Check recent deployments and configuration changes
                - Review historical patterns and similar incidents
                
                4. Mitigation Strategies:
                - Implement quick fixes to reduce customer impact
                - Consider rollback vs. forward fix trade-offs
                - Use traffic shifting for gradual recovery
                - Scale resources if capacity is the issue
                - Implement emergency configuration changes carefully
                
                5. Post-Incident Activities:
                - Conduct blameless post-mortems
                - Identify root causes and contributing factors
                - Create action items with owners and deadlines
                - Share learnings across the organization
                - Update runbooks and documentation
                - Improve monitoring and alerting based on gaps identified
                """
            }
        ]

class RAGSystem:
    """Enhanced RAG system for SRE knowledge"""
    
    def __init__(self, config: RAGConfig, device: str = "cpu"):
        if not RAG_AVAILABLE:
            raise ImportError("RAG dependencies not installed. Run: pip install faiss-cpu sentence-transformers pandas")
        
        self.config = config
        self.device = device
        self.embedding_model = None
        self.vector_db = None
        self.documents = []
        self.metadata = []
        
        self._setup()
    
    def _setup(self):
        """Initialize RAG system"""
        print("🔍 Setting up RAG system...")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(self.config.embedding_model, device=self.device)
        print(f"📝 Embedding model loaded on {self.device}")
        
        # Create directories
        Path(self.config.vector_db_path).mkdir(exist_ok=True)
        Path(self.config.knowledge_base_path).mkdir(exist_ok=True)
        
        # Load or build knowledge base
        self._load_or_build_knowledge_base()
    
    def _chunk_document(self, title: str, content: str) -> List[Dict]:
        """Split document into chunks with metadata"""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), self.config.chunk_size - self.config.chunk_overlap):
            chunk_text = " ".join(words[i:i + self.config.chunk_size])
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text.strip(),
                    'title': title,
                    'chunk_id': f"{title}_chunk_{len(chunks)}",
                    'chunk_index': len(chunks)
                })
        return chunks
    
    def _load_or_build_knowledge_base(self):
        """Load existing or build new knowledge base"""
        vector_file = Path(self.config.vector_db_path) / "vectors.faiss"
        metadata_file = Path(self.config.vector_db_path) / "metadata.pkl"
        
        if vector_file.exists() and metadata_file.exists():
            print("📂 Loading existing vector database...")
            self.vector_db = faiss.read_index(str(vector_file))
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"✅ Loaded vector DB with {len(self.metadata)} chunks")
        else:
            print("🔨 Building new knowledge base...")
            self._build_knowledge_base()
    
    def _build_knowledge_base(self):
        """Build knowledge base from SRE documents"""
        print("📚 Processing SRE documents...")
        
        # Get documents
        sre_docs = SREKnowledgeBase.get_sre_documents()
        
        # Process into chunks
        all_chunks = []
        for doc in sre_docs:
            chunks = self._chunk_document(doc['title'], doc['content'])
            all_chunks.extend(chunks)
        
        print(f"🧩 Created {len(all_chunks)} chunks from {len(sre_docs)} documents")
        
        # Generate embeddings
        print("🧠 Generating embeddings...")
        texts = [chunk['text'] for chunk in all_chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.vector_db = faiss.IndexFlatL2(dimension)
        self.vector_db.add(embeddings.astype('float32'))
        
        self.metadata = all_chunks
        
        # Save to disk
        vector_file = Path(self.config.vector_db_path) / "vectors.faiss"
        metadata_file = Path(self.config.vector_db_path) / "metadata.pkl"
        
        faiss.write_index(self.vector_db, str(vector_file))
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"✅ Knowledge base built with {self.vector_db.ntotal} vectors")
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Search for relevant documents"""
        if top_k is None:
            top_k = self.config.top_k
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search
        scores, indices = self.vector_db.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def generate_rag_prompt(self, query: str, max_context_length: int = 2000) -> str:
        """Generate RAG-enhanced prompt"""
        # Search for relevant context
        context_docs = self.search(query)
        
        # Build context (truncate if needed)
        context_parts = []
        current_length = 0
        
        for doc in context_docs:
            doc_text = f"Source: {doc['title']}\n{doc['text']}\n"
            if current_length + len(doc_text) > max_context_length:
                break
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are an expert Site Reliability Engineer. Use the following context to answer the question comprehensively and practically.

Context:
{context}

Question: {query}

Provide a detailed answer that:
1. References the context where relevant
2. Includes specific troubleshooting steps
3. Mentions relevant tools and commands
4. Follows SRE best practices

Answer:"""
        
        return prompt

class SREFineTuner:
    """Fine-tuning manager with LoRA for SRE scenarios"""
    
    def __init__(self, model_info: ModelInfo):
        if not FINETUNING_AVAILABLE:
            raise ImportError("Fine-tuning dependencies not installed")
        
        self.model_info = model_info
        self.training_data = self._create_training_dataset()
    
    def _create_training_dataset(self) -> List[Dict]:
        """Create SRE-specific training examples"""
        return [
            {
                "instruction": "I have Kubernetes pods stuck in CrashLoopBackOff. Walk me through the troubleshooting process.",
                "response": """Here's a systematic approach to troubleshoot CrashLoopBackOff pods:

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
            }
        ]
    
    def create_dataset(self) -> Dataset:
        """Convert training data to HuggingFace dataset"""
        formatted_data = []
        
        for item in self.training_data:
            # Format for instruction tuning
            text = f"<|user|>\n{item['instruction']}\n<|assistant|>\n{item['response']}<|endoftext|>"
            formatted_data.append({"text": text})
        
        return Dataset.from_list(formatted_data)
    
    def setup_lora_config(self) -> LoraConfig:
        """Setup LoRA configuration"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
    
    def fine_tune(self, output_dir: str = "./sre_fine_tuned") -> str:
        """Fine-tune the model with LoRA"""
        print("🎯 Starting SRE fine-tuning with LoRA...")
        
        # Create dataset
        train_dataset = self.create_dataset()
        print(f"📚 Created dataset with {len(train_dataset)} examples")
        
        # Setup LoRA
        lora_config = self.setup_lora_config()
        peft_model = get_peft_model(self.model_info.model, lora_config)
        
        print("🔧 LoRA configuration applied")
        peft_model.print_trainable_parameters()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=self.model_info.batch_size,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=self.model_info.use_mixed_precision,
            logging_steps=10,
            save_steps=50,
            warmup_steps=10,
            save_total_limit=2,
            remove_unused_columns=False,
            report_to=None,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model_info.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.model_info.tokenizer,
            data_collator=data_collator,
            max_seq_length=512,
        )
        
        # Train
        print("🚀 Starting training...")
        start_time = datetime.now()
        trainer.train()
        training_time = datetime.now() - start_time
        
        print(f"✅ Training completed in {training_time}")
        
        # Save the model
        trainer.save_model()
        print(f"💾 Model saved to {output_dir}")
        
        return output_dir

def test_rag_system(rag: RAGSystem, model_info: ModelInfo):
    """Test RAG system with sample queries"""
    print("\n🧪 Testing RAG System")
    print("=" * 50)
    
    test_queries = [
        "How do I fix Kubernetes pods in CrashLoopBackOff?",
        "What causes 503 errors from load balancers?",
        "How to investigate high memory usage?",
        "Best practices for microservices resilience?"
    ]
    
    for query in test_queries:
        print(f"\n❓ Query: {query}")
        print("-" * 30)
        
        # Get RAG prompt
        rag_prompt = rag.generate_rag_prompt(query)
        
        # Generate response (if model is loaded)
        if model_info.model and model_info.tokenizer:
            try:
                inputs = model_info.tokenizer(rag_prompt, return_tensors="pt", max_length=1024, truncation=True)
                if model_info.device != "cpu":
                    inputs = {k: v.to(model_info.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model_info.model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=model_info.tokenizer.eos_token_id
                    )
                
                response = model_info.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(rag_prompt):].strip()
                print(f"🤖 Response: {response[:300]}...")
                
            except Exception as e:
                print(f"⚠️ Generation failed: {e}")
        else:
            # Just show context retrieval
            context_docs = rag.search(query)
            print(f"📚 Retrieved {len(context_docs)} relevant documents:")
            for i, doc in enumerate(context_docs[:2]):
                print(f"  {i+1}. {doc['title']} (score: {doc['similarity_score']:.3f})")

def main():
    """Main function with CLI arguments"""
    parser = argparse.ArgumentParser(description="SRE RAG & Fine-tuning Pipeline")
    parser.add_argument("--mode", choices=["rag", "finetune", "both"], default="both", 
                       help="Operation mode")
    parser.add_argument("--model-info", type=str, help="Path to serialized model info from main script")
    parser.add_argument("--output-dir", default="./sre_outputs", help="Output directory")
    parser.add_argument("--test", action="store_true", help="Run tests after setup")
    
    args = parser.parse_args()
    
    print("🎯 SRE RAG & Fine-tuning Pipeline")
    print("=" * 50)
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Load model info if provided
    model_info = ModelInfo()
    if args.model_info and Path(args.model_info).exists():
        print(f"📂 Loading model info from {args.model_info}")
        with open(args.model_info, 'rb') as f:
            loaded_info = pickle.load(f)
            model_info = ModelInfo(**loaded_info)
    else:
        print("⚠️ No model info provided - RAG will work, but no generation testing")
    
    # Setup RAG system
    if args.mode in ["rag", "both"]:
        print("\n🔍 Setting up RAG system...")
        rag_config = RAGConfig()
        device = model_info.device if model_info.device != "cpu" else "cpu"
        rag = RAGSystem(rag_config, device=device)
        
        if args.test:
            test_rag_system(rag, model_info)
    
    # Fine-tuning
    if args.mode in ["finetune", "both"] and model_info.model is not None:
        print("\n🎯 Starting fine-tuning...")
        fine_tuner = SREFineTuner(model_info)
        output_path = fine_tuner.fine_tune(f"{args.output_dir}/fine_tuned_model")
        print(f"✅ Fine-tuning completed: {output_path}")
    elif args.mode in ["finetune", "both"]:
        print("⚠️ Fine-tuning requires model info with loaded model")
    
    print(f"\n🎉 Pipeline completed! Check {args.output_dir} for outputs")

if __name__ == "__main__":
    main()