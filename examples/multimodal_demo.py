"""
Multimodal RAG System - Advanced Examples
Demonstrates image, audio, video, and cross-modal capabilities
"""

import asyncio
import sys
sys.path.append('..')

from agentic_rag import MultimodalAgenticRAG
from agentic_rag.multimodal import Modality, MultimodalDocument
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalExamples:
    """Advanced multimodal examples"""
    
    def __init__(self):
        self.rag = None
    
    async def setup(self):
        """Initialize the system"""
        logger.info("Initializing Multimodal RAG System...")
        self.rag = MultimodalAgenticRAG(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            device="cuda",
            quantization="4bit"
        )
        logger.info("System initialized!")
    
    async def example_image_understanding(self):
        """Image understanding and visual QA"""
        
        print("Image Understanding Example")
        
        # Create sample images for demo
        # Red image
        red_image = Image.new('RGB', (224, 224), color='red')
        await self.rag.add_multimodal_document(
            red_image,
            Modality.IMAGE,
            metadata={"color": "red", "type": "solid"}
        )
        
        # Blue image
        blue_image = Image.new('RGB', (224, 224), color='blue')
        await self.rag.add_multimodal_document(
            blue_image,
            Modality.IMAGE,
            metadata={"color": "blue", "type": "solid"}
        )
        
        # Query about images
        result = await self.rag.multimodal_query(
            ("What colors are in the images?", red_image),
            modalities=[Modality.IMAGE],
            use_agent=True
        )
        
        print(f"Query: What colors are in the images?")
        print(f"Answer: {result.answer}")
        print(f"Processing time: {result.processing_time:.2f}s")
    
    async def example_cross_modal_search(self):
        """Cross-modal search between text and images"""
  
        print("Cross-Modal Search Example")

        
        # Add text descriptions
        descriptions = [
            "A beautiful sunset with orange and pink sky",
            "A serene blue ocean with white waves",
            "A green forest with tall trees"
        ]
        
        for desc in descriptions:
            await self.rag.add_multimodal_document(
                desc,
                Modality.TEXT,
                metadata={"type": "description"}
            )
        
        # Create corresponding color images
        colors = [
            ('orange', (255, 165, 0)),
            ('blue', (0, 0, 255)),
            ('green', (0, 255, 0))
        ]
        
        for name, rgb in colors:
            img = Image.new('RGB', (224, 224), color=rgb)
            await self.rag.add_multimodal_document(
                img,
                Modality.IMAGE,
                metadata={"color": name}
            )
        
        # Search for images using text
        result = await self.rag.multimodal_query(
            "Find sunset images",
            modalities=[Modality.IMAGE, Modality.TEXT],
            use_agent=False
        )
        
        print(f"Text->Image Search: 'Find sunset images'")
        print(f"Found {len(result.sources)} matches")
        print(f"Modalities: {[m.value for m in result.modalities_used]}")
    
    async def example_multimodal_reasoning(self):
        """Complex multimodal reasoning"""

        print("Multimodal Reasoning Example")

        
        # Add diverse content
        await self.rag.add_multimodal_document(
            "Climate change is causing global temperatures to rise.",
            Modality.TEXT,
            metadata={"topic": "climate"}
        )
        
        await self.rag.add_multimodal_document(
            "Renewable energy sources include solar and wind power.",
            Modality.TEXT,
            metadata={"topic": "energy"}
        )
        
        # Create a simple chart image (gradient representing temperature rise)
        chart = Image.new('RGB', (224, 224))
        pixels = chart.load()
        for i in range(224):
            for j in range(224):
                # Create gradient from blue to red (cold to hot)
                r = min(255, j)
                b = max(0, 255 - j)
                pixels[i, j] = (r, 0, b)
        
        await self.rag.add_multimodal_document(
            chart,
            Modality.IMAGE,
            metadata={"type": "chart", "topic": "temperature"}
        )
        
        # Complex reasoning query
        result = await self.rag.multimodal_query(
            "Analyze the relationship between the visual data and text information about climate",
            modalities=[Modality.TEXT, Modality.IMAGE],
            use_agent=True
        )
        
        print(f"Complex Query: Climate analysis")
        print(f"Answer: {result.answer[:300]}...")
        
        if result.reasoning_trace:
            print("\nReasoning Steps:")
            for i, step in enumerate(result.reasoning_trace[:3], 1):
                print(f"  Step {i}: {step.get('action', {}).get('type', 'Unknown')}")
    
    async def example_video_processing(self):
        """Video frame extraction and analysis"""

        print("Video Processing Example")

        
        # Simulate video frames
        frames_data = []
        for i in range(8):
            # Create frames with changing colors (simulating video)
            color_value = int(255 * (i / 7))
            frame = Image.new('RGB', (224, 224), color=(color_value, 0, 255 - color_value))
            frames_data.append(frame)
        
        # Process as video
        print("Processing video frames...")
        
        # Add frames as individual images (simulating video processing)
        for i, frame in enumerate(frames_data):
            await self.rag.add_multimodal_document(
                frame,
                Modality.IMAGE,
                metadata={"frame_number": i, "video_id": "sample_video"}
            )
        
        # Query about the video
        result = await self.rag.multimodal_query(
            "Describe the color changes in the video",
            modalities=[Modality.IMAGE],
            use_agent=True
        )
        
        print(f"Query: Describe the color changes")
        print(f"Answer: {result.answer[:200]}...")
    
    async def example_audio_simulation(self):
        """Audio transcription simulation"""

        print("Audio Processing Example")

        
        # Simulate audio transcription
        transcriptions = [
            "Hello, this is a sample audio recording about artificial intelligence.",
            "Machine learning is revolutionizing many industries.",
            "Voice assistants use natural language processing."
        ]
        
        for i, transcript in enumerate(transcriptions):
            await self.rag.add_multimodal_document(
                transcript,
                Modality.TEXT,
                metadata={"source": "audio", "audio_id": f"audio_{i}"}
            )
        
        # Query about audio content
        result = await self.rag.multimodal_query(
            "What topics were discussed in the audio recordings?",
            modalities=[Modality.TEXT],
            use_agent=True
        )
        
        print(f"Query: Topics in audio recordings")
        print(f"Answer: {result.answer}")
    
    async def example_document_types(self):
        """Different document types"""

        print("Document Types Example")

        
        # Add different types of content
        document_types = [
            ("PDF content about quantum computing", "pdf"),
            ("Web article on space exploration", "webpage"),
            ("Research paper on neural networks", "paper"),
            ("Presentation slides on data science", "slides")
        ]
        
        for content, doc_type in document_types:
            await self.rag.add_multimodal_document(
                content,
                Modality.TEXT,
                metadata={"document_type": doc_type}
            )
        
        # Query across document types
        result = await self.rag.multimodal_query(
            "What scientific topics are covered in our documents?",
            modalities=[Modality.TEXT],
            use_agent=True
        )
        
        print(f"Query: Scientific topics in documents")
        print(f"Answer: {result.answer[:300]}...")
        print(f"Sources: {len(result.sources)} documents")
    
    async def example_performance_metrics(self):
        """Performance monitoring"""

        print("Performance Metrics Example")

        
        import time
        import torch
        
        queries = [
            "What is artificial intelligence?",
            "Explain machine learning",
            "How does deep learning work?"
        ]
        
        times = []
        
        for query in queries:
            start = time.time()
            result = await self.rag.multimodal_query(
                query,
                modalities=[Modality.TEXT],
                use_agent=False
            )
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"Query: '{query[:30]}...' - Time: {elapsed:.3f}s")
        
        print(f"\nAverage query time: {np.mean(times):.3f}s")
        
        if torch.cuda.is_available():
            print(f"\nGPU Memory:")
            print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"  Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    async def run_all(self):
        """Run all examples"""
        await self.setup()
        
        examples = [
            self.example_image_understanding,
            self.example_cross_modal_search,
            self.example_multimodal_reasoning,
            self.example_video_processing,
            self.example_audio_simulation,
            self.example_document_types,
            self.example_performance_metrics
        ]
        
        for example in examples:
            try:
                await example()
            except Exception as e:
                print(f"Error in {example.__name__}: {e}")
            
            await asyncio.sleep(1)
  


async def main():
    """Main entry point"""
    print("Multimodal RAG System - Advanced Examples")
    
    
    examples = MultimodalExamples()
    await examples.run_all()


if __name__ == "__main__":
    asyncio.run(main())
