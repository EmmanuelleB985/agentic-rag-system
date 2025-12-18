"""
Agent implementations for the RAG system
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.history = []
    
    @abstractmethod
    async def act(self, observation: str, context: List[Any]) -> str:
        """Take an action based on observation"""
        pass
    
    @abstractmethod
    async def plan(self, goal: str) -> List[str]:
        """Create a plan to achieve a goal"""
        pass
    
    def reset(self):
        """Reset agent state"""
        self.history = []


class ReactiveAgent(BaseAgent):
    """Reactive agent that responds to immediate observations"""
    
    def __init__(self, model_name: str, device: str = "cuda", quantization: str = "4bit"):
        super().__init__(model_name, device)
        self._initialize_model(quantization)
        self.tools = {}
    
    def _initialize_model(self, quantization: str):
        """Initialize the language model"""
        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    async def act(self, observation: str, context: List[Any]) -> str:
        """React to observation"""
        prompt = f"""Observation: {observation}

Context: {str(context)[:500]}

Action:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract action
        if "Action:" in response:
            response = response.split("Action:")[-1].strip()
        
        self.history.append({"observation": observation, "action": response})
        
        return response
    
    async def plan(self, goal: str) -> List[str]:
        """Simple planning"""
        prompt = f"""Goal: {goal}

Create a step-by-step plan:
1."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse steps
        steps = []
        for line in response.split("\n"):
            if line.strip() and (line[0].isdigit() or line.startswith("-")):
                steps.append(line.strip())
        
        return steps
    
    def add_tool(self, name: str, tool_func):
        """Add a tool to the agent"""
        self.tools[name] = tool_func
    
    async def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Use a tool"""
        if tool_name in self.tools:
            return await self.tools[tool_name](**kwargs)
        return f"Tool {tool_name} not found"


class PlanningAgent(BaseAgent):
    """Agent that creates and executes plans"""
    
    def __init__(self, model_name: str, device: str = "cuda", quantization: str = "4bit"):
        super().__init__(model_name, device)
        self._initialize_model(quantization)
        self.current_plan = []
        self.plan_progress = 0
    
    def _initialize_model(self, quantization: str):
        """Initialize the model"""
        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    async def plan(self, goal: str) -> List[str]:
        """Create a detailed plan"""
        prompt = f"""Create a detailed plan to achieve this goal: {goal}

Plan (numbered steps):
1."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.6
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse plan
        plan = []
        for line in response.split("\n"):
            if line.strip() and line[0].isdigit():
                plan.append(line.strip())
        
        self.current_plan = plan
        self.plan_progress = 0
        
        return plan
    
    async def act(self, observation: str, context: List[Any]) -> str:
        """Execute next step in plan"""
        if not self.current_plan:
            return "No plan available. Please create a plan first."
        
        if self.plan_progress >= len(self.current_plan):
            return "Plan completed."
        
        current_step = self.current_plan[self.plan_progress]
        
        prompt = f"""Current step: {current_step}
Observation: {observation}
Context: {str(context)[:300]}

How to execute this step:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7
            )
        
        action = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract execution
        if "How to execute this step:" in action:
            action = action.split("How to execute this step:")[-1].strip()
        
        self.plan_progress += 1
        self.history.append({
            "step": current_step,
            "observation": observation,
            "action": action
        })
        
        return action
    
    def replan(self, feedback: str):
        """Adjust plan based on feedback"""
        logger.info(f"Replanning based on feedback: {feedback}")
        # This would involve updating the plan based on feedback
        pass
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current plan progress"""
        return {
            "plan": self.current_plan,
            "progress": self.plan_progress,
            "completed_ratio": self.plan_progress / len(self.current_plan) if self.current_plan else 0,
            "history": self.history
        }


class ReflexiveAgent(ReactiveAgent):
    """Agent that can reflect on its actions"""
    
    def __init__(self, model_name: str, device: str = "cuda", quantization: str = "4bit"):
        super().__init__(model_name, device, quantization)
        self.reflections = []
    
    async def reflect(self) -> str:
        """Reflect on recent actions"""
        if not self.history:
            return "No actions to reflect on."
        
        recent_history = self.history[-5:]  # Last 5 actions
        
        history_str = "\n".join([
            f"Observation: {h['observation']}\nAction: {h['action']}"
            for h in recent_history
        ])
        
        prompt = f"""Review these recent actions:

{history_str}

Reflection on effectiveness and improvements:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7
            )
        
        reflection = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Reflection on effectiveness" in reflection:
            reflection = reflection.split("Reflection on effectiveness")[-1].strip()
        
        self.reflections.append(reflection)
        
        return reflection
    
    async def act_with_reflection(self, observation: str, context: List[Any]) -> str:
        """Act with consideration of past reflections"""
        # Get base action
        action = await self.act(observation, context)
        
        # Consider reflections if available
        if self.reflections:
            last_reflection = self.reflections[-1]
            
            prompt = f"""Previous reflection: {last_reflection}
Current action: {action}

Improved action based on reflection:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7
                )
            
            improved = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "Improved action" in improved:
                improved = improved.split("Improved action")[-1].strip()
                return improved
        
        return action
