#!/usr/bin/env python3
"""
SRE RAG & Fine-tuning Pipeline - With Universal Training Data Loader
Supports: Markdown, JSON, JSONL from training_data directory
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

# Import the universal training loader
try:
    from training_loader import UniversalTrainingLoader
    TRAINING_LOADER_AVAILABLE = True
except ImportError:
    print("⚠️ training_loader.py not found - using fallback hardcoded examples")
    TRAINING_LOADER_AVAILABLE = False

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
    training_data_dir: str = "./training_data"  # Added for universal loader

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
    """SRE Knowledge Management - Now loads from training_data directory"""
    
    def __init__(self, training_data_dir: str = "./training_data"):
        self.training_data_dir = training_data_dir
        self.loader = None
        if TRAINING_LOADER_AVAILABLE:
            self.loader = UniversalTrainingLoader(training_data_dir)
    
    def get_sre_documents(self) -> List[Dict[str, str]]:
        """
        Load documents from training_data directory.
        Falls back to hardcoded examples if directory is empty.
        """
        if self.loader:
            print(f"📂 Loading knowledge base from {self.training_data_dir}")
            training_examples = self.loader.load_all_data()
            
            if training_examples:
                # Convert training examples to document format for RAG
                documents = []
                for i, example in enumerate(training_examples):
                    documents.append({
                        "title": f"Training Doc {i+1}: {example['instruction'][:50]}...",
                        "content": f"Question: {example['instruction']}\n\nAnswer: {example['response']}"
                    })
                
                print(f"✅ Loaded {len(documents)} documents from training data")
                return documents
        
        # Fallback to hardcoded knowledge base
        print("⚠️ Using fallback hardcoded knowledge base")
        return self._get_fallback_documents()
    
    @staticmethod
    def _get_fallback_documents() -> List[Dict[str, str]]:
        """Fallback hardcoded SRE knowledge base"""
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
                """
            }
        ]

class RAGSystem:
    """Enhanced RAG system - uses training_data directory"""
    
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
        """Load existing or build new knowledge base from training_data"""
        vector_file = Path(self.config.vector_db_path) / "vectors.faiss"
        metadata_file = Path(self.config.vector_db_path) / "metadata.pkl"
        
        # Always rebuild if training_data exists and has files
        training_dir = Path(self.config.training_data_dir)
        has_training_files = False
        if training_dir.exists():
            has_training_files = any(training_dir.glob("*.[md|json|jsonl]*"))
        
        # Rebuild if no vectors OR if training data exists
        if not (vector_file.exists() and metadata_file.exists()) or has_training_files:
            print("🔨 Building knowledge base from training_data directory...")
            self._build_knowledge_base()
        else:
            print("📂 Loading existing vector database...")
            self.vector_db = faiss.read_index(str(vector_file))
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"✅ Loaded vector DB with {len(self.metadata)} chunks")
    
    def _build_knowledge_base(self):
        """Build knowledge base from training_data directory"""
        print("📚 Processing documents from training_data...")
        
        # Get documents from training_data directory
        knowledge_base = SREKnowledgeBase(self.config.training_data_dir)
        sre_docs = knowledge_base.get_sre_documents()
        
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
    """Fine-tuning manager - loads from training_data directory"""
    
    def __init__(self, model_info: ModelInfo, training_data_dir: str = "./training_data"):
        if not FINETUNING_AVAILABLE:
            raise ImportError("Fine-tuning dependencies not installed")
        
        self.model_info = model_info
        self.training_data_dir = training_data_dir
        self.training_data = self._load_training_data()
    
    def _load_training_data(self) -> List[Dict]:
        """Load training data from training_data directory"""
        if TRAINING_LOADER_AVAILABLE:
            loader = UniversalTrainingLoader(self.training_data_dir)
            examples = loader.load_all_data()
            
            if examples:
                print(f"✅ Loaded {len(examples)} training examples for fine-tuning")
                return examples
        
        # Fallback
        print("⚠️ Using fallback hardcoded training examples")
        return self._fallback_training_examples()
    
    @staticmethod
    def _fallback_training_examples() -> List[Dict]:
        """Fallback hardcoded training examples"""
        return [
            {
                "instruction": "Kubernetes pods stuck in CrashLoopBackOff. Walk me through troubleshooting.",
                "response": """Systematic troubleshooting approach:

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
   - Image pull failures

3. **Resolution Actions:**
   - Adjust resource limits if needed
   - Fix configuration issues
   - Update probe settings
   - Rollback problematic deployments"""
            },
            {
                "instruction": "Load balancer returning 503 errors. How to diagnose?",
                "response": """Diagnostic approach:

1. **Check Backend Health:**
   ```bash
   kubectl get endpoints <service-name>
   kubectl describe service <service-name>
   ```

2. **Verify Configuration:**
   - Service selector matches pod labels
   - Port configurations correct
   - Test with port-forward

3. **Health Check Analysis:**
   - Review readiness probe configs
   - Test probe endpoints manually
   - Check probe timing"""
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
    parser.add_argument("--training-data", default="./training_data", help="Training data directory")
    parser.add_argument("--test", action="store_true", help="Run tests after setup")
    parser.add_argument("--create-samples", action="store_true", help="Create sample training files")
    
    args = parser.parse_args()
    
    # Create sample training files if requested
    if args.create_samples:
        if TRAINING_LOADER_AVAILABLE:
            loader = UniversalTrainingLoader(args.training_data)
            loader.create_sample_files()
            print(f"\n✅ Sample files created in {args.training_data}")
            print("Add your own files and run the pipeline again!")
            return
        else:
            print("❌ training_loader.py not found. Cannot create samples.")
            return
    
    print("🎯 SRE RAG & Fine-tuning Pipeline (Universal Loader)")
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
    
    # Setup RAG system (now uses training_data directory)
    if args.mode in ["rag", "both"]:
        print("\n🔍 Setting up RAG system...")
        rag_config = RAGConfig(training_data_dir=args.training_data)
        device = model_info.device if model_info.device != "cpu" else "cpu"
        rag = RAGSystem(rag_config, device=device)
        
        if args.test:
            test_rag_system(rag, model_info)
    
    # Fine-tuning (now uses training_data directory)
    if args.mode in ["finetune", "both"] and model_info.model is not None:
        print("\n🎯 Starting fine-tuning...")
        fine_tuner = SREFineTuner(model_info, training_data_dir=args.training_data)
        output_path = fine_tuner.fine_tune(f"{args.output_dir}/fine_tuned_model")
        print(f"✅ Fine-tuning completed: {output_path}")
    elif args.mode in ["finetune", "both"]:
        print("⚠️ Fine-tuning requires model info with loaded model")
    
    print(f"\n🎉 Pipeline completed! Check {args.output_dir} for outputs")
    print(f"📁 Training data loaded from: {args.training_data}")

if __name__ == "__main__":
    main()

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