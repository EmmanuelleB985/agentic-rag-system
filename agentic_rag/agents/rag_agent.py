"""
RAG Agent implementation with autonomous decision-making and tool use.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class AgentAction(Enum):
    """Possible agent actions."""
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    CALCULATE = "calculate"
    WEB_SEARCH = "web_search"
    REFINE = "refine"
    FINISH = "finish"


@dataclass
class AgentStep:
    """Represents a single step in agent reasoning."""
    number: int
    thought: str
    action: AgentAction
    action_input: Any
    observation: str
    confidence: float


class RAGAgent:
    """
    Autonomous agent that decides when and how to use retrieval.
    Implements ReAct-style reasoning with tool use.
    """
    
    def __init__(self, rag_pipeline):
        """
        Initialize the agent with a RAG pipeline.
        
        Args:
            rag_pipeline: The main RAG pipeline instance
        """
        self.rag = rag_pipeline
        self.max_iterations = 5
        self.decision_threshold = 0.7
        self.tools = self._initialize_tools()
        
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize available tools for the agent."""
        tools = {
            "retrieve": self._tool_retrieve,
            "calculate": self._tool_calculate,
            "web_search": self._tool_web_search,
            "check_knowledge": self._tool_check_knowledge,
        }
        return tools
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Main agent execution loop.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with answer, sources, and reasoning trace
        """
        logger.info(f"Agent processing query: {query}")
        
        # Initialize tracking
        reasoning_trace = []
        iteration = 0
        final_answer = None
        sources = []
        total_tokens = 0
        retrieval_time = 0
        
        # Determine if retrieval is needed
        needs_retrieval = self._assess_retrieval_need(query)
        
        while iteration < self.max_iterations and final_answer is None:
            iteration += 1
            
            # Think about next action
            thought, action, action_input = self._think(query, reasoning_trace)
            
            # Execute action
            start_time = time.time()
            observation = self._execute_action(action, action_input)
            
            if action == AgentAction.RETRIEVE:
                retrieval_time += time.time() - start_time
                sources.extend(observation.get("documents", []))
            
            # Record step
            step = AgentStep(
                number=iteration,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=str(observation),
                confidence=self._calculate_confidence(observation)
            )
            reasoning_trace.append(step)
            
            # Check if we should finish
            if action == AgentAction.FINISH or self._should_finish(reasoning_trace):
                final_answer = observation
                break
            
            # Update token count
            total_tokens += len(self.rag.tokenizer.encode(thought + str(observation)))
        
        # Generate final answer if not already done
        if final_answer is None:
            final_answer = self._generate_final_answer(query, reasoning_trace, sources)
        
        return {
            "answer": final_answer,
            "sources": sources[:5],  # Top 5 sources
            "reasoning_trace": [self._step_to_dict(step) for step in reasoning_trace],
            "confidence": self._calculate_overall_confidence(reasoning_trace),
            "tokens_used": total_tokens,
            "retrieval_time": retrieval_time,
            "iterations": iteration
        }
    
    def _assess_retrieval_need(self, query: str) -> float:
        """
        Assess whether retrieval is needed for the query.
        
        Args:
            query: User query
            
        Returns:
            Probability that retrieval is needed (0-1)
        """
        # Use LLM to assess
        prompt = f"""Determine if external information retrieval is needed to answer this query accurately.

Query: {query}

Consider the following:
1. Is this about specific facts, events, or data?
2. Does this require current information?
3. Can this be answered from general knowledge?
4. Does this require multiple sources?

Output "RETRIEVAL_NEEDED" if retrieval is required, or "NO_RETRIEVAL" if not.

Decision:"""

        response = self.rag.generate(prompt, max_length=10)
        
        if "RETRIEVAL_NEEDED" in response:
            return 0.9
        else:
            return 0.3
    
    def _think(self, query: str, trace: List[AgentStep]) -> Tuple[str, AgentAction, Any]:
        """
        Decide on the next action based on current state.
        
        Args:
            query: Original user query
            trace: Previous reasoning steps
            
        Returns:
            Tuple of (thought, action, action_input)
        """
        # Build context from trace
        context = self._build_context(trace)
        
        # Generate reasoning
        prompt = f"""You are an intelligent agent that must answer the following query:

Query: {query}

Previous steps:
{context}

Think step-by-step about what to do next. Tools available:
- RETRIEVE[query]: Search for relevant information
- CALCULATE[expression]: Perform calculations
- WEB_SEARCH[query]: Search the web for current information
- GENERATE[context]: Generate an answer based on available context
- FINISH[answer]: Provide the final answer

Format your response as:
Thought: [Your reasoning about what to do next]
Action: [TOOL_NAME]
Input: [tool input]

Response:"""

        response = self.rag.generate(prompt, max_length=150)
        
        # Parse response
        thought, action, action_input = self._parse_agent_response(response)
        
        return thought, action, action_input
    
    def _parse_agent_response(self, response: str) -> Tuple[str, AgentAction, Any]:
        """Parse the agent's response into components."""
        # Default values
        thought = "Continuing reasoning..."
        action = AgentAction.GENERATE
        action_input = ""
        
        # Extract thought
        thought_match = re.search(r"Thought:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # Extract action
        action_match = re.search(r"Action:\s*(\w+)", response, re.IGNORECASE)
        if action_match:
            action_str = action_match.group(1).upper()
            
            if "RETRIEVE" in action_str:
                action = AgentAction.RETRIEVE
            elif "CALCULATE" in action_str:
                action = AgentAction.CALCULATE
            elif "WEB" in action_str:
                action = AgentAction.WEB_SEARCH
            elif "FINISH" in action_str:
                action = AgentAction.FINISH
            elif "REFINE" in action_str:
                action = AgentAction.REFINE
            else:
                action = AgentAction.GENERATE
        
        # Extract input
        input_match = re.search(r"Input:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if input_match:
            action_input = input_match.group(1).strip()
        
        return thought, action, action_input
    
    def _execute_action(self, action: AgentAction, action_input: Any) -> Any:
        """Execute the specified action with input."""
        logger.debug(f"Executing action: {action} with input: {action_input}")
        
        if action == AgentAction.RETRIEVE:
            return self._tool_retrieve(action_input)
        elif action == AgentAction.CALCULATE:
            return self._tool_calculate(action_input)
        elif action == AgentAction.WEB_SEARCH:
            return self._tool_web_search(action_input)
        elif action == AgentAction.GENERATE:
            return self._tool_generate(action_input)
        elif action == AgentAction.FINISH:
            return action_input
        else:
            return "Action not recognized"
    
    def _tool_retrieve(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant documents."""
        documents = self.rag.retrieve(query, top_k=5)
        
        # Summarize retrieved content
        if documents:
            summary = f"Found {len(documents)} relevant documents. Top results:\n"
            for i, doc in enumerate(documents[:3], 1):
                summary += f"{i}. {doc['text'][:200]}...\n"
        else:
            summary = "No relevant documents found."
        
        return {
            "summary": summary,
            "documents": documents,
            "success": len(documents) > 0
        }
    
    def _tool_calculate(self, expression: str) -> str:
        """Perform mathematical calculations."""
        try:
            # Safe evaluation of mathematical expressions
            # In production, use a proper math parser
            result = eval(expression, {"__builtins__": {}}, {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "len": len, "pow": pow
            })
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    def _tool_web_search(self, query: str) -> str:
        """Simulate web search (implement with actual API in production)."""
        # This is a placeholder - integrate with actual web search API
        return f"Web search results for '{query}': [Simulated results - integrate with real search API]"
    
    def _tool_check_knowledge(self, topic: str) -> str:
        """Check if the model has knowledge about a topic."""
        prompt = f"Do I have detailed knowledge about: {topic}? Answer YES or NO with brief explanation."
        response = self.rag.generate(prompt, max_length=50)
        return response
    
    def _tool_generate(self, context: str) -> str:
        """Generate text based on context."""
        return self.rag.generate(context, max_length=200)
    
    def _build_context(self, trace: List[AgentStep]) -> str:
        """Build context string from reasoning trace."""
        if not trace:
            return "No previous steps."
        
        context = ""
        for step in trace[-3:]:  # Last 3 steps
            context += f"\nStep {step.number}:\n"
            context += f"  Thought: {step.thought}\n"
            context += f"  Action: {step.action.value}\n"
            context += f"  Result: {step.observation[:200]}...\n"
        
        return context
    
    def _should_finish(self, trace: List[AgentStep]) -> bool:
        """Determine if the agent should stop reasoning."""
        if not trace:
            return False
        
        # Check if we have a good answer
        last_step = trace[-1]
        
        # High confidence answer
        if last_step.confidence > 0.85:
            return True
        
        # Repeated actions without progress
        if len(trace) >= 3:
            recent_actions = [step.action for step in trace[-3:]]
            if len(set(recent_actions)) == 1:  # Same action repeated
                return True
        
        return False
    
    def _generate_final_answer(self, query: str, trace: List[AgentStep], sources: List[Dict]) -> str:
        """Generate the final answer based on all gathered information."""
        # Collect all observations
        observations = []
        for step in trace:
            if step.action == AgentAction.RETRIEVE and step.observation:
                observations.append(step.observation)
        
        # Build final prompt
        context = "\n".join(observations) if observations else "No specific context retrieved."
        
        prompt = f"""Based on the following information, provide a comprehensive answer to the query.

Query: {query}

Available Information:
{context}

Instructions:
- Provide a clear, direct answer
- Cite sources when applicable
- Acknowledge any limitations or uncertainties
- Be concise but complete

Answer:"""

        return self.rag.generate(prompt, max_length=300)
    
    def _calculate_confidence(self, observation: Any) -> float:
        """Calculate confidence score for an observation."""
        confidence = 0.5  # Base confidence
        
        if isinstance(observation, dict):
            if observation.get("success"):
                confidence += 0.3
            if observation.get("documents"):
                # Higher confidence with more relevant documents
                num_docs = len(observation.get("documents", []))
                confidence += min(0.2, num_docs * 0.04)
        elif isinstance(observation, str) and len(observation) > 50:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _calculate_overall_confidence(self, trace: List[AgentStep]) -> float:
        """Calculate overall confidence from reasoning trace."""
        if not trace:
            return 0.0
        
        confidences = [step.confidence for step in trace]
        
        # Weighted average with recent steps having more weight
        weights = [0.5 ** i for i in range(len(confidences)-1, -1, -1)]
        weighted_sum = sum(c * w for c, w in zip(confidences, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _step_to_dict(self, step: AgentStep) -> Dict[str, Any]:
        """Convert AgentStep to dictionary."""
        return {
            "number": step.number,
            "thought": step.thought,
            "action": step.action.value,
            "action_input": str(step.action_input),
            "observation": step.observation[:500],  # Truncate for readability
            "confidence": step.confidence
        }
