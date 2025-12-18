"""
Multimodal Extensions for Agentic RAG System
Support for text, images, audio, and video
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json
import asyncio
from datetime import datetime
import hashlib

# Multimodal imports
from transformers import (
    CLIPModel,
    CLIPProcessor,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
)
from PIL import Image
import librosa
import cv2
import torch.nn.functional as F
import faiss
import chromadb
from chromadb.config import Settings

from .core import AgenticRAG, Document, RetrievalResult

logger = logging.getLogger(__name__)


class Modality(Enum):
    """Supported modalities"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


class DocumentType(Enum):
    """Document types"""
    PDF = "pdf"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    WEBPAGE = "webpage"
    MARKDOWN = "markdown"


@dataclass
class MultimodalDocument:
    """Container for multimodal documents"""
    id: str
    content: Union[str, np.ndarray, bytes]
    modality: Modality
    metadata: Dict[str, Any]
    embeddings: Optional[np.ndarray] = None
    timestamp: Optional[datetime] = None
    source_path: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.id is None:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        content_str = str(self.content)[:100] if isinstance(self.content, str) else str(self.modality)
        return hashlib.md5(f"{content_str}{self.timestamp}".encode()).hexdigest()


@dataclass
class QueryResult:
    """Result from multimodal query"""
    answer: str
    sources: List[MultimodalDocument]
    modalities_used: List[Modality]
    confidence: float
    reasoning_trace: List[Dict[str, Any]]
    visual_outputs: Optional[List[Image.Image]] = None
    audio_outputs: Optional[List[np.ndarray]] = None
    processing_time: Optional[float] = None


class MultimodalEmbedder:
    """Handles embedding generation for different modalities"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding models"""
        from sentence_transformers import SentenceTransformer
        from transformers import pipeline
        
        # Text embeddings
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        
        # Vision-Language model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Audio embeddings
        self.audio_pipeline = pipeline(
            "feature-extraction",
            model="facebook/wav2vec2-base",
            device=0 if self.device == "cuda" else -1
        )
    
    def embed_text(self, texts: List[str]) -> np.ndarray:
        """Generate text embeddings"""
        return self.text_model.encode(texts, convert_to_numpy=True)
    
    def embed_image(self, images: List[Union[str, Image.Image]]) -> np.ndarray:
        """Generate image embeddings"""
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img)
            processed_images.append(img)
        
        inputs = self.clip_processor(images=processed_images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            embeddings = image_features.cpu().numpy()
        
        return embeddings
    
    def embed_audio(self, audio_paths: List[str], sample_rate: int = 16000) -> np.ndarray:
        """Generate audio embeddings"""
        embeddings = []
        
        for audio_path in audio_paths:
            waveform, sr = librosa.load(audio_path, sr=sample_rate)
            features = self.audio_pipeline(waveform)
            embedding = np.mean(features[0], axis=0)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def embed_video(self, video_paths: List[str], frames_per_video: int = 8) -> np.ndarray:
        """Generate video embeddings"""
        all_embeddings = []
        
        for video_path in video_paths:
            frames = self._extract_frames(video_path, frames_per_video)
            frame_embeddings = self.embed_image(frames)
            video_embedding = np.mean(frame_embeddings, axis=0)
            all_embeddings.append(video_embedding)
        
        return np.array(all_embeddings)
    
    def _extract_frames(self, video_path: str, num_frames: int) -> List[Image.Image]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        
        cap.release()
        return frames


class MultimodalRetriever:
    """Handles multimodal document retrieval"""
    
    def __init__(self, embedding_dim: int = 512, use_hybrid_search: bool = True):
        self.embedding_dim = embedding_dim
        self.use_hybrid_search = use_hybrid_search
        self._initialize_indices()
        self._initialize_chroma()
    
    def _initialize_indices(self):
        """Initialize FAISS indices"""
        self.indices = {
            Modality.TEXT: faiss.IndexFlatIP(self.embedding_dim),
            Modality.IMAGE: faiss.IndexFlatIP(self.embedding_dim),
            Modality.AUDIO: faiss.IndexFlatIP(self.embedding_dim),
            Modality.VIDEO: faiss.IndexFlatIP(self.embedding_dim),
            Modality.MULTIMODAL: faiss.IndexFlatIP(self.embedding_dim * 2),
        }
        self.document_stores = {modality: [] for modality in Modality}
    
    def _initialize_chroma(self):
        """Initialize ChromaDB"""
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./multimodal_chroma_db"
        ))
        
        self.collections = {}
        for modality in Modality:
            collection_name = f"multimodal_{modality.value}"
            try:
                self.collections[modality] = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"modality": modality.value}
                )
            except:
                self.collections[modality] = self.chroma_client.get_collection(collection_name)
    
    def add_documents(self, documents: List[MultimodalDocument], modality: Modality):
        """Add documents to retrieval system"""
        if not documents:
            return
        
        embeddings = np.array([doc.embeddings for doc in documents])
        self.indices[modality].add(embeddings)
        self.document_stores[modality].extend(documents)
        
        # Add to ChromaDB
        self.collections[modality].add(
            embeddings=[doc.embeddings.tolist() for doc in documents],
            documents=[doc.content if isinstance(doc.content, str) else str(doc.id) for doc in documents],
            metadatas=[doc.metadata for doc in documents],
            ids=[doc.id for doc in documents]
        )
    
    def search(
        self,
        query_embedding: np.ndarray,
        modality: Modality,
        top_k: int = 5
    ) -> List[Tuple[MultimodalDocument, float]]:
        """Search for similar documents"""
        if modality not in self.indices:
            return []
        
        distances, indices = self.indices[modality].search(
            query_embedding.reshape(1, -1),
            min(top_k, len(self.document_stores[modality]))
        )
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.document_stores[modality]):
                doc = self.document_stores[modality][idx]
                results.append((doc, float(dist)))
        
        return results
    
    def cross_modal_search(
        self,
        query_embeddings: Dict[Modality, np.ndarray],
        top_k: int = 5
    ) -> List[Tuple[MultimodalDocument, float]]:
        """Cross-modal search"""
        all_results = []
        
        for modality, embedding in query_embeddings.items():
            results = self.search(embedding, modality, top_k)
            all_results.extend(results)
        
        # Sort and deduplicate
        all_results = sorted(all_results, key=lambda x: x[1], reverse=True)
        seen_ids = set()
        unique_results = []
        
        for doc, score in all_results:
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                unique_results.append((doc, score))
                if len(unique_results) >= top_k:
                    break
        
        return unique_results


class MultimodalAgent:
    """Agentic component for multimodal reasoning"""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str = "cuda",
        quantization: str = "4bit"
    ):
        self.device = device
        self.model_name = model_name
        self._initialize_models(quantization)
        self.tools = self._setup_tools()
    
    def _initialize_models(self, quantization: str):
        """Initialize models"""
        from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
        
        # Quantization config
        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Language model
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Vision-language model
        self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # Audio model
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-small"
        ).to(self.device)
    
    def _setup_tools(self) -> Dict[str, Any]:
        """Setup available tools"""
        return {
            "retrieve": self._tool_retrieve,
            "analyze_image": self._tool_analyze_image,
            "transcribe_audio": self._tool_transcribe_audio,
            "summarize": self._tool_summarize,
            "compare": self._tool_compare
        }
    
    async def reason(
        self,
        query: str,
        context: List[MultimodalDocument],
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """Multi-step reasoning"""
        reasoning_trace = []
        current_context = context
        
        for iteration in range(max_iterations):
            action = await self._decide_action(query, current_context, reasoning_trace)
            
            if action["type"] == "final_answer":
                return {
                    "answer": action["content"],
                    "reasoning_trace": reasoning_trace,
                    "iterations": iteration + 1
                }
            
            if action["tool"] in self.tools:
                result = await self.tools[action["tool"]](action["parameters"])
                reasoning_trace.append({
                    "iteration": iteration,
                    "action": action,
                    "result": result
                })
                
                if isinstance(result, list):
                    current_context.extend(result)
        
        final_answer = await self._generate_answer(query, current_context, reasoning_trace)
        return {
            "answer": final_answer,
            "reasoning_trace": reasoning_trace,
            "iterations": max_iterations
        }
    
    async def _decide_action(
        self,
        query: str,
        context: List[MultimodalDocument],
        trace: List[Dict]
    ) -> Dict[str, Any]:
        """Decide next action"""
        prompt = self._build_decision_prompt(query, context, trace)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_action(response)
    
    def _build_decision_prompt(
        self,
        query: str,
        context: List[MultimodalDocument],
        trace: List[Dict]
    ) -> str:
        """Build prompt for decision"""
        prompt = f"""Query: {query}

Current Context:
"""
        for doc in context[:3]:
            prompt += f"- {doc.modality.value}: {str(doc.content)[:100]}...\n"
        
        if trace:
            prompt += "\nPrevious Actions:\n"
            for step in trace[-3:]:
                prompt += f"- {step['action']['type']}: {step.get('result', 'pending')[:100]}...\n"
        
        prompt += """
Tools: retrieve, analyze_image, transcribe_audio, summarize, compare, final_answer

Next action:"""
        return prompt
    
    def _parse_action(self, response: str) -> Dict[str, Any]:
        """Parse action from response"""
        response = response.lower()
        
        if "final_answer:" in response:
            answer_start = response.find("final_answer:") + len("final_answer:")
            return {
                "type": "final_answer",
                "content": response[answer_start:].strip()
            }
        
        for tool in self.tools:
            if tool in response:
                return {
                    "type": "tool_use",
                    "tool": tool,
                    "parameters": {"query": response[:100]}
                }
        
        return {
            "type": "tool_use",
            "tool": "retrieve",
            "parameters": {"query": response[:100]}
        }
    
    async def _tool_retrieve(self, params: Dict) -> List[MultimodalDocument]:
        """Retrieve documents"""
        return []
    
    async def _tool_analyze_image(self, params: Dict) -> str:
        """Analyze image"""
        image_path = params.get("image_path", "")
        question = params.get("question", "What is in this image?")
        
        if not image_path:
            return "No image provided"
        
        image = Image.open(image_path)
        inputs = self.blip_processor(image, question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.blip_model.generate(**inputs, max_new_tokens=100)
        
        return self.blip_processor.decode(outputs[0], skip_special_tokens=True)
    
    async def _tool_transcribe_audio(self, params: Dict) -> str:
        """Transcribe audio"""
        audio_path = params.get("audio_path", "")
        
        if not audio_path:
            return "No audio provided"
        
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
        inputs = self.whisper_processor(audio_array, sampling_rate=16000, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(inputs.input_features)
        
        transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0]
    
    async def _tool_summarize(self, params: Dict) -> str:
        """Summarize content"""
        content = params.get("content", "")
        
        inputs = self.tokenizer(
            f"Summarize: {content[:1000]}",
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=100)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    async def _tool_compare(self, params: Dict) -> str:
        """Compare items"""
        items = params.get("items", [])
        
        comparison_prompt = f"Compare: {', '.join(items[:3])}"
        inputs = self.tokenizer(comparison_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=200)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    async def _generate_answer(
        self,
        query: str,
        context: List[MultimodalDocument],
        trace: List[Dict]
    ) -> str:
        """Generate final answer"""
        prompt = f"""Query: {query}

Context:
"""
        for doc in context[:5]:
            prompt += f"- {doc.modality.value}: {str(doc.content)[:200]}...\n"
        
        prompt += "\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=512, temperature=0.7)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class MultimodalAgenticRAG(AgenticRAG):
    """Multimodal Agentic RAG System"""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: str = "cuda",
        quantization: str = "4bit",
        config_path: Optional[str] = None
    ):
        # Initialize base RAG
        super().__init__(
            model_name=model_name,
            device=device,
            quantization=quantization
        )
        
        self.device = device
        
        # Load config
        self.config = self._load_config(config_path) if config_path else self._default_config()
        
        # Initialize multimodal components
        logger.info("Initializing Multimodal components...")
        self.mm_embedder = MultimodalEmbedder(device=device)
        self.mm_retriever = MultimodalRetriever(embedding_dim=512)
        self.mm_agent = MultimodalAgent(model_name, device, quantization)
        
        logger.info("Multimodal Agentic RAG System ready!")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "multimodal": {
                "process_images": True,
                "process_audio": True,
                "process_video": True,
                "cross_modal_retrieval": True
            }
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith('.yaml'):
                import yaml
                return yaml.safe_load(f)
        return self._default_config()
    
    async def add_multimodal_document(
        self,
        content: Union[str, bytes, Image.Image, np.ndarray],
        modality: Modality,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add multimodal document"""
        metadata = metadata or {}
        
        if modality == Modality.TEXT:
            embeddings = self.mm_embedder.embed_text([content] if isinstance(content, str) else [""])
        elif modality == Modality.IMAGE:
            embeddings = self.mm_embedder.embed_image([content])
        elif modality == Modality.AUDIO:
            embeddings = self.mm_embedder.embed_audio([content] if isinstance(content, str) else [""])
        elif modality == Modality.VIDEO:
            embeddings = self.mm_embedder.embed_video([content] if isinstance(content, str) else [""])
        else:
            return "Unsupported modality"
        
        doc = MultimodalDocument(
            id=hashlib.md5(str(content).encode()).hexdigest(),
            content=content,
            modality=modality,
            metadata=metadata,
            embeddings=embeddings[0] if embeddings.size > 0 else np.zeros(512)
        )
        
        self.mm_retriever.add_documents([doc], modality)
        return f"Added {modality.value} document"
    
    async def multimodal_query(
        self,
        query: Union[str, Image.Image, Tuple[str, Image.Image]],
        modalities: Optional[List[Modality]] = None,
        use_agent: bool = True,
        top_k: int = 5
    ) -> QueryResult:
        """Process multimodal query"""
        import time
        start_time = time.time()
        
        # Generate embeddings
        query_embeddings = {}
        
        if isinstance(query, str):
            query_text = query
            query_embeddings[Modality.TEXT] = self.mm_embedder.embed_text([query])[0]
        elif isinstance(query, Image.Image):
            query_text = "Image query"
            query_embeddings[Modality.IMAGE] = self.mm_embedder.embed_image([query])[0]
        elif isinstance(query, tuple):
            query_text, query_image = query
            query_embeddings[Modality.TEXT] = self.mm_embedder.embed_text([query_text])[0]
            query_embeddings[Modality.IMAGE] = self.mm_embedder.embed_image([query_image])[0]
        else:
            query_text = str(query)
            query_embeddings[Modality.TEXT] = self.mm_embedder.embed_text([query_text])[0]
        
        # Retrieve documents
        retrieved_docs = []
        modalities_to_search = modalities or list(Modality)
        
        for modality in modalities_to_search:
            if modality in query_embeddings:
                docs = self.mm_retriever.search(query_embeddings[modality], modality, top_k)
                retrieved_docs.extend([doc for doc, _ in docs])
        
        # Generate response
        if use_agent:
            result = await self.mm_agent.reason(query_text, retrieved_docs)
            answer = result["answer"]
            reasoning_trace = result["reasoning_trace"]
        else:
            answer = await self.mm_agent._generate_answer(query_text, retrieved_docs, [])
            reasoning_trace = []
        
        processing_time = time.time() - start_time
        
        return QueryResult(
            answer=answer,
            sources=retrieved_docs[:top_k],
            modalities_used=[doc.modality for doc in retrieved_docs],
            confidence=0.85,
            reasoning_trace=reasoning_trace,
            processing_time=processing_time
        )
