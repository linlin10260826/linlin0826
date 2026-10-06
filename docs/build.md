# Build notes

The root CMake target builds only the portable YOLO postprocessor. The Efinix reference program is intentionally kept separate because it requires the Ti60F225 BSP, generated SoC headers, TensorFlow Lite Micro sources, and the TinyML accelerator library from the original Efinity project.

Typical host check:

```powershell
cmake -S . -B build
cmake --build build
```

A successful host build checks the postprocessor API only. It does not prove that a board image, camera interface, or HDMI output is configured.
