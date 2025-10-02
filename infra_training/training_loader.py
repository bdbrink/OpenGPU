#!/usr/bin/env python3
"""
Universal Training Data Loader for SRE RAG & Fine-tuning
Supports: Markdown (.md), JSON (.json), JSONL (.jsonl)
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional

class UniversalTrainingLoader:
    """Load training data from multiple file formats"""
    
    def __init__(self, data_dir: str = "./training_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    # ============ JSON LOADERS ============
    
    def load_json_file(self, filepath: Path) -> List[Dict]:
        """
        Load JSON training data.
        
        Supported formats:
        1. Array of objects: [{"instruction": "...", "response": "..."}, ...]
        2. Object with examples key: {"examples": [{"instruction": "...", "response": "..."}, ...]}
        3. Q&A format: [{"question": "...", "answer": "..."}, ...]
        """
        examples = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            # Direct list of examples
            for item in data:
                example = self._normalize_json_item(item)
                if example:
                    examples.append(example)
        elif isinstance(data, dict):
            # Check for common keys
            for key in ['examples', 'data', 'training_data', 'conversations']:
                if key in data and isinstance(data[key], list):
                    for item in data[key]:
                        example = self._normalize_json_item(item)
                        if example:
                            examples.append(example)
                    break
            else:
                # Try to treat the dict itself as a single example
                example = self._normalize_json_item(data)
                if example:
                    examples.append(example)
        
        return examples
    
    def load_jsonl_file(self, filepath: Path) -> List[Dict]:
        """Load JSONL (JSON Lines) format - one JSON object per line"""
        examples = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    example = self._normalize_json_item(item)
                    if example:
                        examples.append(example)
                except json.JSONDecodeError as e:
                    print(f"  Warning: Invalid JSON on line {line_num}: {e}")
        
        return examples
    
    def _normalize_json_item(self, item: Dict) -> Optional[Dict]:
        """Normalize different JSON formats to standard instruction/response format"""
        if not isinstance(item, dict):
            return None
        
        # Try common field name variations
        instruction_keys = ['instruction', 'input', 'prompt', 'question', 'query', 'user', 'human']
        response_keys = ['response', 'output', 'answer', 'completion', 'assistant', 'ai']
        
        instruction = None
        response = None
        
        # Find instruction
        for key in instruction_keys:
            if key in item:
                instruction = item[key]
                break
        
        # Find response
        for key in response_keys:
            if key in item:
                response = item[key]
                break
        
        # Handle conversation format (messages array)
        if 'messages' in item and isinstance(item['messages'], list):
            for msg in item['messages']:
                if isinstance(msg, dict):
                    role = msg.get('role', '').lower()
                    content = msg.get('content', '')
                    if role in ['user', 'human'] and not instruction:
                        instruction = content
                    elif role in ['assistant', 'ai'] and not response:
                        response = content
        
        if instruction and response:
            return {
                'instruction': str(instruction).strip(),
                'response': str(response).strip()
            }
        
        return None
    
    # ============ MARKDOWN LOADERS ============
    
    def load_markdown_file(self, filepath: Path, format_type: str = "auto") -> List[Dict]:
        """Load markdown with auto-format detection"""
        if format_type == "auto":
            return self._auto_detect_markdown_format(filepath)
        elif format_type == "sections":
            return self._parse_section_format(filepath)
        elif format_type == "qa":
            return self._parse_qa_format(filepath)
        elif format_type == "conversational":
            return self._parse_conversational_format(filepath)
        return []
    
    def _auto_detect_markdown_format(self, filepath: Path) -> List[Dict]:
        """Auto-detect markdown format"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for Q&A format
        if re.search(r'##?\s*Q:', content, re.IGNORECASE):
            return self._parse_qa_format(filepath)
        
        # Check for conversational format
        if re.search(r'\*?\*?(User|Human):\*?\*?', content, re.IGNORECASE):
            return self._parse_conversational_format(filepath)
        
        # Default to section format
        return self._parse_section_format(filepath)
    
    def _parse_section_format(self, filepath: Path) -> List[Dict]:
        """
        Parse section-based markdown:
        # Question
        Answer...
        ---
        # Next Question
        Answer...
        """
        examples = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by horizontal rules or headers
        sections = re.split(r'\n---+\n|\n(?=^# )', content, flags=re.MULTILINE)
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Split into instruction and response
            lines = section.split('\n', 1)
            if len(lines) < 2:
                continue
            
            instruction = lines[0].strip('#').strip()
            response = lines[1].strip() if len(lines) > 1 else ""
            
            if instruction and response:
                examples.append({
                    "instruction": instruction,
                    "response": response
                })
        
        return examples
    
    def _parse_qa_format(self, filepath: Path) -> List[Dict]:
        """Parse Q&A markdown format"""
        examples = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Match Q&A pairs
        pattern = r'##?\s*Q:\s*(.+?)\n+\*?\*?A:\*?\*?\s*(.+?)(?=\n##?\s*Q:|\Z)'
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        
        for question, answer in matches:
            examples.append({
                "instruction": question.strip(),
                "response": answer.strip()
            })
        
        return examples
    
    def _parse_conversational_format(self, filepath: Path) -> List[Dict]:
        """Parse conversational markdown format"""
        examples = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Match User/Assistant or Human/AI pairs
        pattern = r'\*?\*?(User|Human):\*?\*?\s*(.+?)\n+\*?\*?(Assistant|AI):\*?\*?\s*(.+?)(?=\n\*?\*?(User|Human):|\Z)'
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            user_msg = match[1].strip()
            assistant_msg = match[3].strip()
            if user_msg and assistant_msg:
                examples.append({
                    "instruction": user_msg,
                    "response": assistant_msg
                })
        
        return examples
    
    # ============ MAIN LOADER ============
    
    def load_all_data(self) -> List[Dict]:
        """
        Load ALL training data from the directory.
        Supports: .md, .json, .jsonl files
        """
        all_examples = []
        
        if not self.data_dir.exists():
            print(f"Warning: Training data directory '{self.data_dir}' doesn't exist")
            print(f"Creating directory: {self.data_dir}")
            self.data_dir.mkdir(parents=True)
            return all_examples
        
        # Find all supported files
        supported_extensions = ['.md', '.json', '.jsonl']
        all_files = []
        for ext in supported_extensions:
            all_files.extend(self.data_dir.glob(f"*{ext}"))
        
        if not all_files:
            print(f"Warning: No training files found in '{self.data_dir}'")
            print(f"Supported formats: {', '.join(supported_extensions)}")
            return all_examples
        
        print(f"\nFound {len(all_files)} training data files")
        print("=" * 50)
        
        # Load each file
        for filepath in sorted(all_files):
            print(f"\nLoading: {filepath.name}")
            
            try:
                if filepath.suffix == '.json':
                    examples = self.load_json_file(filepath)
                elif filepath.suffix == '.jsonl':
                    examples = self.load_jsonl_file(filepath)
                elif filepath.suffix == '.md':
                    examples = self.load_markdown_file(filepath, format_type="auto")
                else:
                    continue
                
                all_examples.extend(examples)
                print(f"  Loaded {len(examples)} examples")
                
            except Exception as e:
                print(f"  Error loading {filepath.name}: {e}")
        
        print(f"\n{'='*50}")
        print(f"Total examples loaded: {len(all_examples)}")
        print(f"{'='*50}\n")
        
        return all_examples
    
    # ============ SAMPLE FILE GENERATOR ============
    
    def create_sample_files(self):
        """Create sample files in all supported formats"""
        print(f"Creating sample training files in: {self.data_dir}")
        
        # Sample 1: Markdown section format
        md_section = """# Kubernetes pods stuck in CrashLoopBackOff. Walk me through troubleshooting.

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
"""
        
        # Sample 2: Markdown Q&A format
        md_qa = """## Q: How do I implement circuit breakers in microservices?

**A:** Circuit breaker implementation:

1. Choose a library (Resilience4j, Hystrix, pybreaker)
2. Configure failure thresholds
3. Set up monitoring
4. Implement fallback mechanisms

## Q: Best practices for monitoring Kubernetes?

**A:** Monitoring best practices:

1. Use Prometheus for metrics
2. Set up Grafana dashboards
3. Configure alerts based on SLIs/SLOs
4. Implement distributed tracing
"""
        
        # Sample 3: JSON array format
        json_array = [
            {
                "instruction": "What's the first step when investigating high CPU usage?",
                "response": "Start by identifying which process is consuming CPU:\n1. Run `top` or `htop`\n2. Check `kubectl top pods` for container CPU\n3. Review application metrics\n4. Analyze historical trends"
            },
            {
                "question": "How to debug DNS issues in Kubernetes?",
                "answer": "DNS debugging steps:\n1. Test with `nslookup` from a pod\n2. Check CoreDNS logs\n3. Verify service endpoints\n4. Test external DNS resolution"
            }
        ]
        
        # Sample 4: JSONL format
        jsonl_lines = [
            {"instruction": "Explain database connection pooling", "response": "Connection pooling reuses database connections to improve performance:\n- Reduces connection overhead\n- Configure pool size based on load\n- Monitor idle vs active connections\n- Set appropriate timeouts"},
            {"prompt": "What causes memory leaks?", "completion": "Common memory leak causes:\n1. Unreleased references\n2. Event listener accumulation\n3. Cache without eviction\n4. Circular references\n5. Static collections growing unbounded"}
        ]
        
        # Sample 5: JSON with nested structure
        json_nested = {
            "training_data": [
                {
                    "instruction": "How to handle cascading failures?",
                    "response": "Cascading failure prevention:\n1. Implement circuit breakers\n2. Use timeouts and retries\n3. Apply rate limiting\n4. Isolate critical services\n5. Monitor dependency health"
                }
            ],
            "metadata": {
                "version": "1.0",
                "topic": "SRE Best Practices"
            }
        }
        
        # Sample 6: Conversational format
        md_conversation = """**User:** Application is slow. Where do I start?

**Assistant:** Start with these diagnostics:
1. Check application logs for errors
2. Monitor response times and latency
3. Review resource utilization (CPU, memory, disk I/O)
4. Check database query performance
5. Analyze network connectivity

**User:** How do I optimize database queries?

**Assistant:** Query optimization techniques:
1. Use EXPLAIN to analyze query plans
2. Add appropriate indexes
3. Avoid N+1 query problems
4. Use connection pooling
5. Cache frequently accessed data
"""
        
        # Write all sample files
        samples = [
            ("sre_k8s_troubleshooting.md", md_section),
            ("sre_architecture_qa.md", md_qa),
            ("sre_basic_training.json", json.dumps(json_array, indent=2)),
            ("sre_debug_tips.jsonl", "\n".join(json.dumps(item) for item in jsonl_lines)),
            ("sre_advanced.json", json.dumps(json_nested, indent=2)),
            ("sre_conversations.md", md_conversation)
        ]
        
        for filename, content in samples:
            filepath = self.data_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  Created: {filepath}")
        
        print(f"\nSample files created! Run your training to load them.")


# Example usage
if __name__ == "__main__":
    # Create sample files
    loader = UniversalTrainingLoader("./training_data")
    
    print("Creating sample training files...")
    loader.create_sample_files()
    
    print("\n\nLoading all training data...")
    examples = loader.load_all_data()
    
    # Show some examples
    if examples:
        print("\nExample loaded data:")
        for i, example in enumerate(examples[:3], 1):
            print(f"\n--- Example {i} ---")
            print(f"Instruction: {example['instruction'][:80]}...")
            print(f"Response: {example['response'][:80]}...")