"""Classifier component for AI-powered file classification using Ollama."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ..analyzer.analyzer import AnalysisResult
from ..config.models import AISettings, ConfidenceThresholds
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationResult:
    """Result of file classification."""

    # File identification
    file_path: Path
    filename: str

    # Classification outputs
    destination_folder: str
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""

    # Status
    success: bool = True
    error_message: str | None = None

    @property
    def confidence_level(self) -> str:
        """Get confidence level as a string."""
        if self.confidence >= 0.9:
            return "high"
        elif self.confidence >= 0.6:
            return "medium"
        else:
            return "low"


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5:latest",
        temperature: float = 0.1,
        max_retries: int = 3,
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama API base URL
            model_name: Model to use for classification
            temperature: LLM temperature (lower = more deterministic)
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = httpx.Timeout(120.0, connect=10.0)  # Long timeout for LLM

    def _check_connection(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama connection check failed: {e}")
            return False

    def _check_model_available(self) -> bool:
        """Check if the configured model is available."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    # Check for exact match or base model name
                    base_name = self.model_name.split(":")[0]
                    return any(
                        self.model_name in m or base_name in m for m in models
                    )
        except Exception as e:
            logger.warning(f"Model availability check failed: {e}")
        return False

    def generate(self, prompt: str) -> str | None:
        """
        Generate a response from Ollama.

        Args:
            prompt: The prompt to send to the model

        Returns:
            Generated text response or None if failed
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        return data.get("response", "")

                    logger.warning(
                        f"Ollama returned status {response.status_code} "
                        f"on attempt {attempt + 1}/{self.max_retries}"
                    )
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"

            except httpx.TimeoutException as e:
                logger.warning(f"Ollama timeout on attempt {attempt + 1}: {e}")
                last_error = f"Timeout: {e}"

            except httpx.RequestError as e:
                logger.warning(f"Ollama request error on attempt {attempt + 1}: {e}")
                last_error = f"Request error: {e}"

        logger.error(f"All {self.max_retries} attempts to Ollama failed: {last_error}")
        return None


class FileClassifier:
    """
    Classifies files using local LLM via Ollama.

    Uses file content and metadata to determine:
    - Destination folder
    - Tags
    - Confidence score
    """

    CLASSIFICATION_PROMPT_TEMPLATE = '''You are a file organization assistant. Analyze this file and suggest where it should be stored.

FILE INFORMATION:
- Filename: {filename}
- Extension: {extension}
- Size: {size} bytes
- Created: {created}
- Modified: {modified}

FILE CONTENT (preview):
{content_preview}

Based on this information, determine:
1. The most appropriate destination folder (use a logical folder structure like "Documents/Work", "Projects/Personal", "Finances/Receipts", etc.)
2. Relevant tags for this file
3. Your confidence in this classification (0.0 to 1.0)
4. Brief reasoning for your decision

Respond ONLY with valid JSON in this exact format (no other text):
{{
    "destination_folder": "Category/Subcategory",
    "tags": ["tag1", "tag2", "tag3"],
    "confidence": 0.85,
    "reasoning": "Brief explanation of why this classification was chosen"
}}'''

    def __init__(
        self,
        ai_settings: AISettings | None = None,
        confidence_thresholds: ConfidenceThresholds | None = None,
    ):
        """
        Initialize the file classifier.

        Args:
            ai_settings: AI configuration settings
            confidence_thresholds: Threshold settings for confidence levels
        """
        self.ai_settings = ai_settings or AISettings()
        self.confidence_thresholds = confidence_thresholds or ConfidenceThresholds()

        self.ollama = OllamaClient(
            base_url=self.ai_settings.ollama_base_url,
            model_name=self.ai_settings.model_name,
            temperature=self.ai_settings.temperature,
            max_retries=self.ai_settings.max_retries,
        )

    def check_ollama_status(self) -> tuple[bool, str]:
        """
        Check if Ollama is ready for classification.

        Returns:
            Tuple of (is_ready, status_message)
        """
        if not self.ollama._check_connection():
            return False, f"Cannot connect to Ollama at {self.ai_settings.ollama_base_url}"

        if not self.ollama._check_model_available():
            return False, f"Model '{self.ai_settings.model_name}' not found. Run: ollama pull {self.ai_settings.model_name}"

        return True, "Ollama is ready"

    def _build_prompt(self, analysis: AnalysisResult) -> str:
        """Build the classification prompt from analysis result."""
        return self.CLASSIFICATION_PROMPT_TEMPLATE.format(
            filename=analysis.metadata.filename,
            extension=analysis.metadata.extension,
            size=analysis.metadata.size_bytes,
            created=analysis.metadata.created_at.strftime("%Y-%m-%d %H:%M"),
            modified=analysis.metadata.modified_at.strftime("%Y-%m-%d %H:%M"),
            content_preview=analysis.content_preview[:2000],  # Limit context size
        )

    def _parse_response(self, response: str, file_path: Path) -> ClassificationResult:
        """Parse LLM response into ClassificationResult."""
        try:
            # Try to extract JSON from response (LLM might add extra text)
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in response")

            data = json.loads(json_match.group())

            # Validate required fields
            destination = data.get("destination_folder", "Unsorted")
            tags = data.get("tags", [])
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "No reasoning provided")

            # Sanitize destination folder (remove leading/trailing slashes, etc.)
            destination = destination.strip("/\\").replace("\\", "/")
            if not destination:
                destination = "Unsorted"

            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))

            # Ensure tags is a list of strings
            if not isinstance(tags, list):
                tags = [str(tags)] if tags else []
            tags = [str(t).strip() for t in tags if t]

            return ClassificationResult(
                file_path=file_path,
                filename=file_path.name,
                destination_folder=destination,
                tags=tags,
                confidence=confidence,
                reasoning=reasoning,
                success=True,
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}\nResponse was: {response[:500]}")
            return ClassificationResult(
                file_path=file_path,
                filename=file_path.name,
                destination_folder="Unsorted",
                tags=[],
                confidence=0.0,
                reasoning="",
                success=False,
                error_message=f"Failed to parse LLM response: {e}",
            )

    def classify(self, analysis: AnalysisResult) -> ClassificationResult:
        """
        Classify a file based on its analysis results.

        Args:
            analysis: AnalysisResult from the analyzer component

        Returns:
            ClassificationResult with suggested destination and tags
        """
        # Handle failed analysis
        if not analysis.success:
            return ClassificationResult(
                file_path=analysis.file_path,
                filename=analysis.file_path.name,
                destination_folder="Unsorted",
                tags=[],
                confidence=0.0,
                reasoning="",
                success=False,
                error_message=f"Analysis failed: {analysis.error_message}",
            )

        # Build prompt and get classification
        prompt = self._build_prompt(analysis)

        logger.info(f"Classifying {analysis.file_path.name} with {self.ai_settings.model_name}")

        response = self.ollama.generate(prompt)

        if response is None:
            return ClassificationResult(
                file_path=analysis.file_path,
                filename=analysis.file_path.name,
                destination_folder="Unsorted",
                tags=[],
                confidence=0.0,
                reasoning="",
                success=False,
                error_message="Failed to get response from Ollama",
            )

        result = self._parse_response(response, analysis.file_path)

        if result.success:
            logger.info(
                f"Classified {analysis.file_path.name}: "
                f"destination='{result.destination_folder}', "
                f"confidence={result.confidence:.2f}, "
                f"tags={result.tags}"
            )

        return result

    def classify_multiple(
        self, analyses: list[AnalysisResult]
    ) -> list[ClassificationResult]:
        """
        Classify multiple files.

        Args:
            analyses: List of AnalysisResult objects

        Returns:
            List of ClassificationResult objects
        """
        results: list[ClassificationResult] = []
        for analysis in analyses:
            results.append(self.classify(analysis))
        return results
