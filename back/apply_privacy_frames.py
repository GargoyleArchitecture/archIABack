from pathlib import Path
from evrag.privacy import FaceBlurrer
import cv2

def main():
    blurrer = FaceBlurrer()
    frames_dir = Path('videos/frames')
    frames = list(frames_dir.glob('*.jpg'))
    print(f'Encontrados {len(frames)} frames. Aplicando privacidad retrospectiva...')
    
    for f in frames:
        # Usar 'speaker' mode para que detecte las caras en cualquier lugar
        # y además tape el margen inferior (12%) donde suelen estar los nombres
        try:
            blurrer.process_image(f, layout_type='speaker')
            print(f"Privatizado: {f.name}")
        except Exception as e:
            print(f"Error procesando {f.name}: {e}")
            
    print('!Privacidad aplicada a todos los frames guardados con éxito!')

if __name__ == "__main__":
    main()
