#!/usr/bin/env python3
"""
Intelligent SRE Assistant - Natural workflow with baseline, triage, and learning
"""

import torch
import pickle
import os
import sys
import warnings
import subprocess
import re
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '11.0.0'


class KnowledgeBase:
    """Persistent knowledge base for learning from interactions"""
    
    def __init__(self, kb_path: str = "./sre_knowledge.json"):
        self.kb_path = Path(kb_path)
        self.data = self._load()
    
    def _load(self) -> Dict:
        """Load knowledge base from disk"""
        if self.kb_path.exists():
            try:
                with open(self.kb_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            "cluster_baseline": {},
            "known_issues": [],
            "resolutions": [],
            "patterns": [],
            "last_health_check": None
        }
    
    def save(self):
        """Save knowledge base to disk"""
        try:
            with open(self.kb_path, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save knowledge base: {e}")
    
    def update_baseline(self, key: str, value: any):
        """Update cluster baseline"""
        self.data["cluster_baseline"][key] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        self.save()
    
    def add_issue(self, issue: str, severity: str, context: str):
        """Record an issue"""
        self.data["known_issues"].append({
            "issue": issue,
            "severity": severity,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "resolved": False
        })
        self.save()
    
    def add_resolution(self, issue: str, solution: str, commands: List[str]):
        """Record a successful resolution"""
        self.data["resolutions"].append({
            "issue": issue,
            "solution": solution,
            "commands": commands,
            "timestamp": datetime.now().isoformat()
        })
        self.save()
    
    def get_baseline_summary(self) -> str:
        """Get human-readable baseline summary"""
        if not self.data["cluster_baseline"]:
            return "No baseline established yet."
        
        lines = ["Cluster Baseline:"]
        for key, info in self.data["cluster_baseline"].items():
            lines.append(f"  • {key}: {info['value']}")
        return "\n".join(lines)


class OutputFormatter:
    """Clean and format terminal output"""
    
    @staticmethod
    def format_command_result(command: str, result: Dict) -> str:
        """Format command execution result cleanly"""
        if result['success']:
            output = result['stdout'].strip()
            if len(output) > 2000:
                output = output[:2000] + "\n... (output truncated)"
            return f"\n💻 {command}\n{output}\n"
        else:
            stderr = result.get('stderr', 'Unknown error').strip()
            if len(stderr) > 500:
                stderr = stderr[:500] + "..."
            return f"\n❌ {command}\nError: {stderr}\n"
    
    @staticmethod
    def clean_response(text: str) -> str:
        """Clean up model response for professional output"""
        # Remove duplicate code blocks
        text = re.sub(r'```\s*```', '', text)
        text = re.sub(r'```\s*$', '', text)
        text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
        
        # Remove filler phrases
        fillers = [
            r"Let's run that\.?\s*",
            r"Let me run that\.?\s*", 
            r"I'll execute that\.?\s*",
            r"Running that now\.?\s*",
            r"Here's what I found\.?\s*",
        ]
        for pattern in fillers:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Remove duplicate command echoes
        text = re.sub(r'```bash\s*\$?\s*\w+[^\n]*\n```\s*(?=💻)', '', text)
        
        # Clean up excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'```[a-z]*\s*```', '', text)
        
        return text.strip()
    
    @staticmethod
    def format_assistant_response(text: str) -> str:
        """Format assistant response with nice structure"""
        lines = text.split('\n')
        formatted = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code = not in_code
                continue
            if in_code:
                continue
            if line.strip():
                formatted.append(line)
        
        return '\n'.join(formatted)


class CommandExecutor:
    """Safely executes shell commands with allowlist"""
    
    def __init__(self, allowed_commands: Optional[List[str]] = None):
        self.allowed_commands = allowed_commands or [
            'kubectl', 'docker', 'helm', 'git', 'ls', 'cat', 'grep', 
            'ps', 'df', 'du', 'top', 'netstat', 'curl', 'ping',
            'systemctl', 'journalctl', 'free', 'uptime', 'whoami',
            'aws', 'gcloud', 'az'
        ]
        self.timeout = 30
    
    def is_allowed(self, command: str) -> Tuple[bool, str]:
        """Check if command is in allowlist"""
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return False, "Empty command"
        
        base_cmd = cmd_parts[0]
        if base_cmd not in self.allowed_commands:
            return False, f"Command '{base_cmd}' not in allowlist"
        
        dangerous = ['rm', 'delete', 'drop', 'truncate', '>', '>>', 'sudo', 'su']
        for danger in dangerous:
            if danger in command.lower():
                return False, f"Dangerous operation: {danger}"
        
        return True, "Allowed"
    
    def execute(self, command: str) -> Dict:
        """Execute command and return output"""
        allowed, reason = self.is_allowed(command)
        
        if not allowed:
            return {
                'success': False,
                'error': reason,
                'stdout': '',
                'stderr': reason
            }
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            return {
                'success': result.returncode == 0,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': command
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f'Timeout after {self.timeout}s',
                'stdout': '',
                'stderr': 'Timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'stdout': '',
                'stderr': str(e)
            }


class SREAssistant:
    """Intelligent SRE assistant with natural workflow"""
    
    def __init__(self, model_info_path: str, auto_approve: bool = False):
        """Load model and initialize"""
        print("🔄 Loading SRE assistant...")
        
        with open(model_info_path, 'rb') as f:
            info = pickle.load(f)
        
        self.tokenizer = info['tokenizer']
        self.model = info['model']
        self.device = info['device']
        self.model_id = info.get('model_id', 'Unknown Model')
        
        # Components
        self.auto_approve = auto_approve
        self.executor = CommandExecutor()
        self.knowledge = KnowledgeBase()
        self.history: List[Dict] = []
        
        # State tracking
        self.baseline_established = bool(self.knowledge.data["cluster_baseline"])
        self.session_start = datetime.now()
        
        print(f"✅ {self.model_id}")
        print(f"📍 Device: {self.device}")
        print(f"🧠 Knowledge base: {'loaded' if self.baseline_established else 'new'}")
        if auto_approve:
            print("⚡ Auto-approve: ON")
        print()
    
    def _ask_permission(self, command: str) -> bool:
        """Ask user for permission to run command"""
        if self.auto_approve:
            return True
        
        print(f"\n💭 Run: {command}")
        while True:
            try:
                response = input("   [y/n/always]: ").strip().lower()
                if response in ['y', 'yes']:
                    return True
                elif response in ['n', 'no']:
                    return False
                elif response in ['a', 'always']:
                    self.auto_approve = True
                    print("   ✅ Auto-approve enabled")
                    return True
            except (KeyboardInterrupt, EOFError):
                print("\n   ⛔ Denied")
                return False
    
    def _build_context_prompt(self, user_input: str) -> str:
        """Build intelligent context-aware prompt"""
        
        # Core context
        context = [
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Working dir: {os.getcwd()}"
        ]
        
        # Add knowledge base context
        if self.baseline_established:
            context.append("\nKnown baseline:")
            context.append(self.knowledge.get_baseline_summary())
        
        # Recent patterns
        if self.knowledge.data["known_issues"]:
            recent = [i for i in self.knowledge.data["known_issues"] if not i.get("resolved")]
            if recent:
                context.append(f"\nOpen issues: {len(recent)}")
        
        context_str = "\n".join(context)
        
        # Build intelligent prompt based on state
        if not self.baseline_established:
            mode = "BASELINE_MODE"
            instructions = """You're establishing a cluster baseline. Gather key metrics:
- Node count, versions, resources (CPU, memory)
- Pod distribution and health
- Key namespaces and workloads
- Network and storage status

Use [EXEC:command] to run commands. Be methodical and thorough."""
        
        else:
            mode = "OPERATIONAL_MODE"
            instructions = """You're an experienced SRE monitoring this cluster. 
- Check for anomalies vs baseline
- Investigate issues systematically
- Suggest actionable remediation
- Learn from patterns

Use [EXEC:command] when you need data. Be proactive but not alarmist."""
        
        prompt = f"""You are an experienced Site Reliability Engineer.

CONTEXT:
{context_str}

MODE: {mode}
{instructions}

Available commands: kubectl, docker, curl, grep, ps, df, top, etc.

USER: {user_input}

ASSISTANT:"""
        
        return prompt
    
    def _process_commands(self, text: str) -> str:
        """Process embedded commands"""
        exec_pattern = r'\[EXEC:\s*([^\]]+?)\]'
        commands = re.findall(exec_pattern, text)
        
        # Deduplicate
        unique_cmds = []
        seen = set()
        for cmd in commands:
            clean = cmd.strip()
            if clean not in seen:
                unique_cmds.append(clean)
                seen.add(clean)
        
        # Execute commands
        results = {}
        for command in unique_cmds:
            if not self._ask_permission(command):
                results[command] = f"\n⚠️  Command declined: {command}\n"
                continue
            
            result = self.executor.execute(command)
            results[command] = OutputFormatter.format_command_result(command, result)
        
        # Replace in text
        def replace(match):
            cmd = match.group(1).strip()
            return results.get(cmd, '')
        
        text = re.sub(exec_pattern, replace, text)
        return OutputFormatter.clean_response(text)
    
    def _extract_insights(self, response: str, user_input: str):
        """Extract and learn from interactions"""
        
        # Detect if establishing baseline
        if not self.baseline_established and "kubectl get nodes" in response.lower():
            # Look for node count in response
            node_match = re.search(r'(\d+)\s+(?:nodes?|Ready)', response, re.IGNORECASE)
            if node_match:
                count = node_match.group(1)
                self.knowledge.update_baseline("node_count", count)
                print(f"\n📊 Baseline updated: {count} nodes")
        
        # Detect issues
        issue_patterns = [
            (r'crashloopbackoff', 'high', 'Pod crash loop detected'),
            (r'notready|not ready', 'high', 'Node not ready'),
            (r'pending', 'medium', 'Pending pods detected'),
            (r'evicted', 'medium', 'Pod evictions occurred'),
        ]
        
        for pattern, severity, description in issue_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                self.knowledge.add_issue(description, severity, user_input)
    
    def generate_response(self, user_input: str, max_tokens: int = 800) -> str:
        """Generate intelligent response"""
        
        # Build context-aware prompt
        prompt = self._build_context_prompt(user_input)
        
        # Add conversation history for context
        if self.history:
            history_context = []
            for msg in self.history[-2:]:
                history_context.append(f"User: {msg['user'][:100]}")
                history_context.append(f"Assistant: {msg['assistant'][:150]}")
            full_prompt = "\n".join(history_context) + "\n\n" + prompt
        else:
            full_prompt = prompt
        
        # Tokenize
        inputs = self.tokenizer(
            full_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        gen_config = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
        }
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_config)
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(full_prompt):].strip()
            
            # Process commands
            response = self._process_commands(response)
            
            # Stop at conversation boundaries
            for marker in ['\n\nUser:', '\nYou:']:
                idx = response.find(marker)
                if idx > 100:
                    response = response[:idx].strip()
                    break
            
            # Learn from interaction
            self._extract_insights(response, user_input)
            
            # Update history
            self.history.append({
                "user": user_input,
                "assistant": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Check if baseline is now established
            if not self.baseline_established and len(self.knowledge.data["cluster_baseline"]) >= 3:
                self.baseline_established = True
                print("\n✨ Cluster baseline established! Now in operational mode.\n")
            
            return response
            
        except Exception as e:
            return f"❌ Error: {e}"
    
    def chat(self):
        """Main chat loop"""
        print("🤖 SRE Assistant")
        print("=" * 60)
        
        if not self.baseline_established:
            print("💡 First time? I'll help establish a cluster baseline.")
            print("   Try: 'give me a cluster overview'\n")
        else:
            print("💡 Baseline loaded. Ready for operations.")
            print("   Try: 'check cluster health' or 'investigate pod issues'\n")
        
        print("Commands: /quit, /baseline, /issues, /history, /save, /auto")
        print("=" * 60)
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle meta commands
                if user_input.startswith('/'):
                    if user_input in ['/quit', '/exit']:
                        print("\n👋 Goodbye!")
                        break
                    
                    elif user_input == '/baseline':
                        print("\n" + self.knowledge.get_baseline_summary())
                        continue
                    
                    elif user_input == '/issues':
                        issues = [i for i in self.knowledge.data["known_issues"] if not i.get("resolved")]
                        if issues:
                            print(f"\n🔴 Open issues: {len(issues)}")
                            for i, issue in enumerate(issues[-5:], 1):
                                print(f"  {i}. [{issue['severity']}] {issue['issue']}")
                        else:
                            print("\n✅ No open issues")
                        continue
                    
                    elif user_input == '/history':
                        if self.history:
                            print(f"\n📝 Last {min(5, len(self.history))} interactions:")
                            for msg in self.history[-5:]:
                                print(f"\n• {msg['user'][:60]}...")
                        else:
                            print("\n📝 No history yet")
                        continue
                    
                    elif user_input == '/auto':
                        self.auto_approve = not self.auto_approve
                        status = "ON" if self.auto_approve else "OFF"
                        print(f"✅ Auto-approve: {status}")
                        continue
                    
                    elif user_input.startswith('/save'):
                        filename = f"sre_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        with open(filename, 'w') as f:
                            for msg in self.history:
                                f.write(f"You: {msg['user']}\n")
                                f.write(f"Assistant: {msg['assistant']}\n\n")
                        print(f"💾 Saved to {filename}")
                        continue
                    
                    else:
                        print(f"❌ Unknown command: {user_input}")
                        continue
                
                # Generate response
                print("\n🤖 ", end="", flush=True)
                response = self.generate_response(user_input)
                formatted = OutputFormatter.format_assistant_response(response)
                print(formatted)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()


def find_models(models_dir: str = "./models") -> List[Path]:
    """Find cached model files"""
    path = Path(models_dir)
    if not path.exists():
        return []
    
    pkl_files = list(path.glob("*/model_info.pkl"))
    root = path / "model_info.pkl"
    if root.exists():
        pkl_files.append(root)
    
    return pkl_files


def select_model() -> Optional[Path]:
    """Interactive model selection"""
    models = find_models()
    
    if not models:
        print("❌ No models found in ./models")
        print("💡 Run training script first")
        return None
    
    if len(models) == 1:
        print(f"📦 Using: {models[0].parent.name}")
        return models[0]
    
    print(f"\n📦 Found {len(models)} models:\n")
    for i, m in enumerate(models, 1):
        name = m.parent.name if m.parent.name != "models" else "root"
        mod_time = datetime.fromtimestamp(m.stat().st_mtime)
        print(f"  {i}. {name} ({mod_time.strftime('%Y-%m-%d %H:%M')})")
    
    while True:
        try:
            choice = input(f"\nSelect (1-{len(models)}) or 'q': ").strip()
            if choice.lower() == 'q':
                return None
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    return models[idx]
        except (KeyboardInterrupt, EOFError):
            return None


def main():
    """Main entry point"""
    print("🚀 SRE Assistant - Intelligent Operations")
    print("=" * 60)
    print()
    
    model_path = select_model()
    if not model_path:
        print("❌ No model selected")
        return
    
    print()
    
    try:
        assistant = SREAssistant(
            str(model_path),
            auto_approve=False  # Change to True to skip confirmations
        )
        assistant.chat()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()