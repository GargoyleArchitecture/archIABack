"""
Privacy & Anonymization Module for EVRAG - Production Ready
"""

import re
import json
import datetime
import subprocess
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple


def get_win_path(p):
    if os.name != 'nt': return str(p)
    abs_p = os.path.abspath(str(p))
    if not abs_p.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_p
    return abs_p


@dataclass
class AnonymizationResult:
    original_text: str
    anonymized_text: str
    entities_removed: list[str] = field(default_factory=list)
    faces_blurred: int = 0
    regions_hidden: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "anonymized_text": self.anonymized_text,
            "entities_removed": self.faces_blurred,
            "regions_hidden": self.regions_hidden,
        }


class TextAnonymizer:
    def __init__(self, language: str = "es"):
        self.language = language
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def anonymize(self, text: str, remove_dates: bool = False) -> AnonymizationResult:
        entities_removed = []
        anonymized_text = text
        patterns = {
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',
            r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b': '[TELEFONO]',
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b': '[TARJETA]'
        }
        for pattern, replacement in patterns.items():
            found = re.findall(pattern, anonymized_text)
            for item in found: entities_removed.append(f"{item}")
            anonymized_text = re.sub(pattern, replacement, anonymized_text)
            
        # IA LLM Extraction + Regex
        system_prompt = (
            "Eres un experto en privacidad de datos. Tu tarea es extraer TODOS los nombres "
            "completos y de pila de personas mencionados en la siguiente transcripción. "
            "No incluyas nombres de empresas, tecnologías ni lugares. "
            "Devuelve ÚNICAMENTE un arreglo JSON de strings con los nombres encontrados. "
            "Si no hay nombres, devuelve un arreglo vacío []. No agregues markdown ni explicaciones."
        )
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=anonymized_text)
            ])
            content = response.content.strip()
            if content.startswith("```json"): content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"): content = content.replace("```", "").strip()
            
            names_to_redact = json.loads(content)
            names_to_redact = [n for n in names_to_redact if len(n) > 2]
            
            for name in names_to_redact:
                entities_removed.append(f"{name} (NAME)")
                pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
                anonymized_text = pattern.sub("[REDACTADO]", anonymized_text)
        except Exception as e:
            print(f"[Privacy] Error en extracción LLM: {e}")
            
        return AnonymizationResult(text, anonymized_text, entities_removed)


class ImageAnonymizer:
    def __init__(self):
        self.face_cascade = None
        self._cascade_loaded = False

    def _load_cascade(self):
        if self._cascade_loaded: return
        import cv2
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self._cascade_loaded = True
        
    def blur_frames_batch(self, frames: list[Path]) -> list[Tuple[Path, int]]:
        results = []
        for frame in frames:
            # Simple heuristic for layout_type based on index, for now just default to slide
            res = self.process_image(frame, layout_type="slide")
            results.append((frame, res.faces_blurred))
        return results

    def process_image(
        self, 
        image_path: Path | str, 
        output_path: Path | None = None,
        layout_type: str = "slide" 
    ) -> AnonymizationResult:
        import cv2
        import numpy as np
        import stat

        win_image_path = get_win_path(image_path)
        image = None
        for attempt in range(3):
            try:
                os.chmod(win_image_path, stat.S_IWRITE)
                with open(win_image_path, 'rb') as f:
                    chunk = np.frombuffer(f.read(), dtype=np.uint8)
                    image = cv2.imdecode(chunk, cv2.IMREAD_COLOR)
                if image is not None: break
            except: time.sleep(0.3)
        
        if image is None: raise ValueError(f"Falla lectura: {win_image_path}")
        
        h, w = image.shape[:2]
        faces_count = 0
        regions_hidden = 0

        # RECTÁNGULO NEGRO PURO (solicitado por usuario)
        mask_color = (0, 0, 0) 

        if layout_type == "gallery":
            image[:] = mask_color
            cv2.putText(image, "CONTENIDO PRIVADO (GALERIA)", (w//10, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            regions_hidden = 1

        elif layout_type == "slide":
            # Lateral derecho (Participantes en Zoom/Teams) - 25%
            rw = int(w * 0.25)
            image[0:h, w-rw:w] = mask_color
            
            # Pie de página (Nombres / CC / Controles) - 15%
            bh = int(h * 0.15)
            image[h-bh:h, 0:w] = mask_color
            regions_hidden = 2
        else:
            self._load_cascade()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            for (x, y, fw, fh) in faces:
                cv2.rectangle(image, (x, y), (x+fw, y+fh), mask_color, -1)
                faces_count += 1
            # Etiqueta de nombre en Speaker mode - 12%
            lh = int(h * 0.12)
            image[h-lh:h, 0:w] = mask_color
            regions_hidden = 1

        target_p = get_win_path(output_path or image_path)
        _, img_encoded = cv2.imencode('.jpg', image)
        for attempt in range(3):
            try:
                with open(target_p, 'wb') as f:
                    f.write(img_encoded)
                break
            except: time.sleep(0.3)

        return AnonymizationResult("", "", [], faces_count, regions_hidden)


class SecureStorage:
    def __init__(self, storage_path: Path | str):
        self.storage_path = Path(storage_path)

    def verify_bitlocker(self) -> bool:
        try:
            result = subprocess.run(["manage-bde", "-status", str(self.storage_path.drive)], capture_output=True, text=True)
            return "Protection Status:        Protection On" in result.stdout
        except: return False

    def secure_delete_video(self, video_path: Path | str) -> bool:
        p = get_win_path(video_path)
        if not os.path.exists(p): return False
        try:
            file_size = os.path.getsize(p)
            with open(p, 'wb') as f: f.write(b'\x00' * file_size)
            os.remove(p)
            return True
        except: return False

    def log_access(self, user: str, action: str, resource: str):
        log_dir = self.storage_path / ".access_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"access_{datetime.datetime.now().strftime('%Y-%m-%d')}.log"
        with open(get_win_path(log_file), 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] {user} - {action} - {resource}\n")


FaceBlurrer = ImageAnonymizer
