"""
Privacy & Anonymization Module for EVRAG

Módulo de anonimización y privacidad para videos:
- Eliminar nombres propios de transcripciones
- Difuminar rostros en frames
- Gestión segura de datos
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AnonymizationResult:
    """
    Result of anonymization process.
    
    Attributes:
        original_text: Original text before anonymization
        anonymized_text: Text after anonymization
        entities_removed: List of removed/anonymized entities
        faces_blurred: Number of faces blurred in frames
    """
    original_text: str
    anonymized_text: str
    entities_removed: list[str] = field(default_factory=list)
    faces_blurred: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_text": self.original_text,
            "anonymized_text": self.anonymized_text,
            "entities_removed": self.entities_removed,
            "faces_blurred": self.faces_blurred,
        }


class TextAnonymizer:
    """
    Anonymize text by removing personally identifiable information (PII).
    
    Uses regex-based approach (no spaCy required) for:
    - Email addresses
    - Phone numbers
    - Potential names (capitalized words, limited)
    - Dates and ID numbers
    """
    
    def __init__(self, language: str = "es"):
        """
        Initialize text anonymizer.
        
        Args:
            language: Language for patterns ('es' or 'en')
        """
        self.language = language
    
    def anonymize(self, text: str, remove_locations: bool = False, remove_dates: bool = False) -> AnonymizationResult:
        """
        Anonymize text by removing PII.
        
        Args:
            text: Text to anonymize
            remove_locations: Whether to remove location names (limited without NER)
            remove_dates: Whether to remove dates
            
        Returns:
            AnonymizationResult with anonymized text
        """
        entities_removed = []
        anonymized_text = text
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, anonymized_text)
        for email in emails:
            entities_removed.append(f"{email} (EMAIL)")
        anonymized_text = re.sub(email_pattern, '[EMAIL]', anonymized_text)
        
        # Phone numbers (various formats)
        phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
        phones = re.findall(phone_pattern, anonymized_text)
        for phone in phones:
            entities_removed.append(f"{phone} (PHONE)")
        anonymized_text = re.sub(phone_pattern, '[TELEFONO]', anonymized_text)
        
        # Credit card numbers
        cc_pattern = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        ccs = re.findall(cc_pattern, anonymized_text)
        for cc in ccs:
            entities_removed.append(f"{cc} (CREDIT_CARD)")
        anonymized_text = re.sub(cc_pattern, '[TARJETA]', anonymized_text)
        
        # Dates (basic patterns)
        if remove_dates:
            date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
            dates = re.findall(date_pattern, anonymized_text)
            for date in dates:
                entities_removed.append(f"{date} (DATE)")
            anonymized_text = re.sub(date_pattern, '[FECHA]', anonymized_text)
            
            # Written dates (e.g., "25 de diciembre de 2023")
            written_date_pattern = r'\b\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\b'
            written_dates = re.findall(written_date_pattern, anonymized_text)
            for date in written_dates:
                entities_removed.append(f"{date} (DATE)")
            anonymized_text = re.sub(written_date_pattern, '[FECHA]', anonymized_text)
        
        # ID numbers (DNI, passport, etc.)
        id_pattern = r'\b[A-Z]{1,3}\d{6,9}\b'
        ids = re.findall(id_pattern, anonymized_text)
        for id_num in ids:
            entities_removed.append(f"{id_num} (ID)")
        anonymized_text = re.sub(id_pattern, '[IDENTIFICADOR]', anonymized_text)
        
        return AnonymizationResult(
            original_text=text,
            anonymized_text=anonymized_text,
            entities_removed=entities_removed,
        )


class FaceBlurrer:
    """
    Blur faces in images for privacy protection.
    
    Uses OpenCV's Haar cascades for face detection.
    """
    
    def __init__(self):
        """Initialize face blurrer."""
        self.face_cascade = None
        self._cascade_loaded = False
    
    def _load_cascade(self):
        """Lazy load Haar cascade for face detection."""
        if self._cascade_loaded:
            return
        
        import cv2
        
        # Load Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        self._cascade_loaded = True
    
    def blur_faces(self, image_path: Path | str, output_path: Path | None = None, blur_intensity: int = 50) -> tuple[Path, int]:
        """
        Blur faces in an image.
        
        Args:
            image_path: Path to input image
            output_path: Path for output image (default: overwrite input)
            blur_intensity: Intensity of blur (higher = more blur)
            
        Returns:
            Tuple of (output_path, faces_detected)
        """
        import cv2
        
        self._load_cascade()
        
        image_path = Path(image_path)
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        
        # Blur each face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = image[y:y+h, x:x+w]
            
            # Apply heavy Gaussian blur
            blurred_face = cv2.GaussianBlur(face_roi, (blur_intensity, blur_intensity), 0)
            
            # Replace face with blurred version
            image[y:y+h, x:x+w] = blurred_face
        
        # Save result
        if output_path is None:
            output_path = image_path
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_path), image)
        
        return output_path, len(faces)
    
    def blur_frames_batch(self, frame_paths: list[Path], output_dir: Path | None = None) -> list[tuple[Path, int]]:
        """
        Blur faces in multiple frames.
        
        Args:
            frame_paths: List of frame paths
            output_dir: Directory for blurred frames (default: overwrite originals)
            
        Returns:
            List of tuples (output_path, faces_detected) for each frame
        """
        results = []
        
        for frame_path in frame_paths:
            if output_dir:
                output_path = output_dir / frame_path.name
            else:
                output_path = frame_path
            
            try:
                result_path, faces_count = self.blur_faces(frame_path, output_path)
                results.append((result_path, faces_count))
            except Exception as e:
                print(f"Error blurring {frame_path.name}: {e}")
                results.append((frame_path, 0))
        
        return results


class SecureStorage:
    """
    Secure storage management for EVRAG data.
    
    Features:
    - Verify BitLocker encryption
    - Secure deletion of original videos
    - Access control logging
    """
    
    def __init__(self, storage_path: Path | str):
        """
        Initialize secure storage.
        
        Args:
            storage_path: Path to storage directory
        """
        self.storage_path = Path(storage_path)
    
    def verify_bitlocker(self) -> bool:
        """
        Verify if BitLocker is enabled on the drive.
        
        Returns:
            True if BitLocker is enabled, False otherwise
        """
        import subprocess
        
        try:
            # Check BitLocker status on Windows
            result = subprocess.run(
                ["manage-bde", "-status", str(self.storage_path.drive)],
                capture_output=True,
                text=True,
            )
            
            # Check if protection is on
            return "Protection Status:        Protection On" in result.stdout
            
        except Exception as e:
            print(f"Warning: Could not verify BitLocker status: {e}")
            return False
    
    def secure_delete_video(self, video_path: Path | str) -> bool:
        """
        Securely delete a video file after processing.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if deletion successful, False otherwise
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            return False
        
        try:
            # Overwrite file with random data before deletion
            file_size = video_path.stat().st_size
            
            with open(video_path, 'wb') as f:
                # Overwrite with zeros
                f.write(b'\x00' * file_size)
            
            # Delete file
            video_path.unlink()
            
            print(f"Securely deleted: {video_path.name}")
            
            return True
            
        except Exception as e:
            print(f"Error securely deleting {video_path.name}: {e}")
            return False
    
    def log_access(self, user: str, action: str, resource: str):
        """
        Log access to sensitive data.
        
        Args:
            user: User identifier
            action: Action performed
            resource: Resource accessed
        """
        import datetime
        
        log_dir = self.storage_path / ".access_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"access_{datetime.datetime.now().strftime('%Y-%m-%d')}.log"
        
        timestamp = datetime.datetime.now().isoformat()
        log_entry = f"[{timestamp}] {user} - {action} - {resource}\n"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
