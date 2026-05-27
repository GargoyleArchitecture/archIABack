@echo off
REM EVRAG Batch Processor - Windows
REM Procesa todos los videos en back/videos/raw/ con EVRAG

echo ========================================
echo EVRAG Batch Processor
echo ========================================
echo.

cd /d C:\Users\arnul\proyectos\trabajo\Tesis\archIABack

echo Processing videos with EVRAG...
echo.

REM Video 1: Top 15 System Design Patterns
echo [1/5] Top15_System_Design_Patterns.mp4
poetry run python -m back.evrag --video "back/videos/raw/Top15_System_Design_Patterns.mp4" --reprocess
echo.

REM Video 2: Event-Driven Architecture
echo [2/5] Event_Driven_Architecture.mp4
poetry run python -m back.evrag --video "back/videos/raw/Event_Driven_Architecture.mp4" --reprocess
echo.

REM Video 3: Event Sourcing & CQRS
echo [3/5] Event_Sourcing_CQRS.mp4
poetry run python -m back.evrag --video "back/videos/raw/Event_Sourcing_CQRS.mp4" --reprocess
echo.

REM Video 4: Master Software Architecture GOTO 2025
echo [4/5] Master_Software_Architecture_GOTO_2025.mp4
poetry run python -m back.evrag --video "back/videos/raw/Master_Software_Architecture_GOTO_2025.mp4" --reprocess
echo.

REM Video 5: Scalable Resilient Architectures
echo [5/5] Scalable_Resilient_Architectures.mp4
poetry run python -m back.evrag --video "back/videos/raw/Scalable_Resilient_Architectures.mp4" --reprocess
echo.

echo ========================================
echo All videos processed!
echo ========================================
pause
