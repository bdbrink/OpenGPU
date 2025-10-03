#!/usr/bin/env python3
"""
SRE RAG & Fine-tuning Pipeline - Clean Version
Loads training data ONLY from training_data directory
Supports: Markdown, JSON, JSONL formats
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
    print("❌ training_loader.py not found - this script requires it to run")
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
    training_data_dir: str = "./training_data"

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
    """SRE Knowledge Management - Loads ONLY from training_data directory"""
    
    def __init__(self, training_data_dir: str = "./training_data"):
        self.training_data_dir = training_data_dir
        self.loader = None
        
        if not TRAINING_LOADER_AVAILABLE:
            raise ImportError("training_loader.py is required but not found")
        
        self.loader = UniversalTrainingLoader(training_data_dir)
    
    def get_sre_documents(self) -> List[Dict[str, str]]:
        """
        Load documents from training_data directory.
        Returns empty list if no data found.
        """
        print(f"📂 Loading knowledge base from {self.training_data_dir}")
        training_examples = self.loader.load_all_data()
        
        if not training_examples:
            print(f"⚠️ No training data found in {self.training_data_dir}")
            print("   Add .md, .json, or .jsonl files to proceed")
            return []
        
        # Convert training examples to document format for RAG
        documents = []
        for i, example in enumerate(training_examples):
            documents.append({
                "title": f"Training Doc {i+1}: {example['instruction'][:50]}...",
                "content": f"Question: {example['instruction']}\n\nAnswer: {example['response']}"
            })
        
        print(f"✅ Loaded {len(documents)} documents from training data")
        return documents

class RAGSystem:
    """Enhanced RAG system - uses training_data directory only"""
    
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
        
        if not sre_docs:
            print("❌ No documents to process. Add training data files and try again.")
            raise ValueError("No training data available")
        
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
    """Fine-tuning manager - loads ONLY from training_data directory"""
    
    def __init__(self, model_info: ModelInfo, training_data_dir: str = "./training_data"):
        if not FINETUNING_AVAILABLE:
            raise ImportError("Fine-tuning dependencies not installed")
        
        if not TRAINING_LOADER_AVAILABLE:
            raise ImportError("training_loader.py is required but not found")
        
        self.model_info = model_info
        self.training_data_dir = training_data_dir
        self.training_data = self._load_training_data()
    
    def _load_training_data(self) -> List[Dict]:
        """Load training data from training_data directory"""
        loader = UniversalTrainingLoader(self.training_data_dir)
        examples = loader.load_all_data()
        
        if not examples:
            raise ValueError(f"No training data found in {self.training_data_dir}")
        
        print(f"✅ Loaded {len(examples)} training examples for fine-tuning")
        return examples
    
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
        
        # Ensure tokenizer has pad token
        if self.model_info.tokenizer.pad_token is None:
            self.model_info.tokenizer.pad_token = self.model_info.tokenizer.eos_token
        
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
        
        # Trainer - tokenizer passed differently in newer TRL versions
        try:
            # Try newer TRL API (tokenizer as part of model)
            trainer = SFTTrainer(
                model=peft_model,
                args=training_args,
                train_dataset=train_dataset,
                dataset_text_field="text",
                max_seq_length=512,
                tokenizer=self.model_info.tokenizer,
                packing=False,
            )
        except TypeError as e:
            # Fallback for older TRL versions or different API
            print("⚠️ Adjusting for TRL version compatibility...")
            from transformers import Trainer
            
            # Use standard Trainer with data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.model_info.tokenizer,
                mlm=False,
            )
            
            # Need to tokenize the dataset first
            def tokenize_function(examples):
                return self.model_info.tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
            
            tokenized_dataset = train_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=train_dataset.column_names
            )
            
            trainer = Trainer(
                model=peft_model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
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
    
    print("🎯 SRE RAG & Fine-tuning Pipeline (Training Data Only)")
    print("=" * 50)
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Check if training data directory exists and has files
    training_dir = Path(args.training_data)
    if not training_dir.exists() or not any(training_dir.glob("*.[md|json|jsonl]*")):
        print(f"\n❌ No training data found in {args.training_data}")
        print("\nOptions:")
        print(f"  1. Run with --create-samples to generate example files")
        print(f"  2. Add your own .md, .json, or .jsonl files to {args.training_data}")
        return
    
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
        rag_config = RAGConfig(training_data_dir=args.training_data)
        device = model_info.device if model_info.device != "cpu" else "cpu"
        
        try:
            rag = RAGSystem(rag_config, device=device)
            
            if args.test:
                test_rag_system(rag, model_info)
        except ValueError as e:
            print(f"❌ RAG setup failed: {e}")
            return
    
    # Fine-tuning
    if args.mode in ["finetune", "both"] and model_info.model is not None:
        print("\n🎯 Starting fine-tuning...")
        try:
            fine_tuner = SREFineTuner(model_info, training_data_dir=args.training_data)
            output_path = fine_tuner.fine_tune(f"{args.output_dir}/fine_tuned_model")
            print(f"✅ Fine-tuning completed: {output_path}")
        except ValueError as e:
            print(f"❌ Fine-tuning failed: {e}")
            return
    elif args.mode in ["finetune", "both"]:
        print("⚠️ Fine-tuning requires model info with loaded model")
    
    print(f"\n🎉 Pipeline completed! Check {args.output_dir} for outputs")
    print(f"📁 Training data loaded from: {args.training_data}")

if __name__ == "__main__":
    main()