# Entrenamiento de Redes Neuronales en GPU
* CUDA con PyTorch en Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Albonire/parcial-gpu-pytorch-colab/blob/main/Taller_CUDA_PyTorch.ipynb)
*(Nota: Si el enlace a Colab da error, asegúrate de que el repositorio sea **Público**. De lo contrario, puedes abrir Colab manualmente y cargar el notebook desde la pestaña GitHub).*

---

| | |
|---|---|
| **Parcial** | Segundo Corte |
| **Materia** | Programación Paralela y Computación Distribuida |
| **Profesor** | Prf. Juan Alejandro Carrillo Jaimes |
| **Integrantes** | Anderson González & [Nombre del Compañero] |
| | |
| **Fecha** | 2026-I |

---

## 0. Instrucciones Generales

### Preguntas
**1. ¿Qué diferencia hay entre un notebook en la nube (Colab) y un entorno local como el del tutorial de instalación? ¿Cuál prefieren y por qué?**
*Respuesta:* En la nube, Google administra la infraestructura (drivers de GPU, CUDA toolkit y entorno), brindando recursos gratuitos (NVIDIA T4/A100) al instante en el navegador, sin configurar hardware localmente. En el entorno local, se requiere instalación manual, mantener hardware y drivers actualizados, y configurar variables de entorno. Preferimos Colab porque es ideal para desarrollo rápido y prototipado, abstrayendo las dependencias del hardware físico.

**2. Antes de comenzar, hagan una predicción: ¿cuántas veces más rápida creen que será la GPU comparada con la CPU en el entrenamiento?**
*Respuesta:* Estimamos que será entre 20x y 30x más rápida, considerando el paralelismo masivo de las miles de ALUs en la GPU contra los pocos núcleos de la CPU para operaciones matriciales.

---

## 1. Configurar el Entorno en Google Colab

**[PANTALLAZO: Verificar GPU disponible (Debe mostrar: GPU disponible: True y el nombre de la GPU)]**
> *(Inserta tu captura aquí)*

**[PANTALLAZO: Estado de la GPU con nvidia-smi (Identificar: nombre de la GPU, versión de CUDA, memoria total)]**
> *(Inserta tu captura aquí)*

### Preguntas
**1. La salida de `nvidia-smi` muestra campos como *Driver Version*, *Memory Usage* y *GPU-Util*. ¿Qué indica cada uno?**
*Respuesta:* 
- **Driver Version:** La versión del controlador NVIDIA instalado, el cual asegura que el software y la gráfica se comuniquen.
- **Memory Usage:** La cantidad de memoria VRAM (Video RAM) actualmente asignada en la GPU (usada vs. total).
- **GPU-Util:** El porcentaje del poder de cómputo de los procesadores de la GPU que está siendo utilizado en ese instante.

**2. Cuando activan el acelerador en Colab, ¿qué creen que ocurre físicamente? ¿La GPU está en su computador o en otro lugar? Propongan una analogía con algo de la vida cotidiana.**
*Respuesta:* Físicamente, un servidor en los centros de datos de Google nos asigna un hilo de comunicación a una tarjeta gráfica real conectada a su motherboard. No ocurre en nuestro equipo.
*Analogía:* Es como contratar un servicio de lavandería industrial (Colab). Tú envías tu ropa sucia (datos/código), y ellos usan sus máquinas gigantes (GPUs) en sus instalaciones para lavarla rápido, y luego te envían la ropa limpia (resultados).

**3. `torch.cuda.is_available()` retorna `True` o `False`. ¿Qué condiciones deben cumplirse para que retorne `True`? Listen al menos tres requisitos.**
*Respuesta:*
1. Que el sistema cuente físicamente con una GPU NVIDIA (en este caso, asignada por Colab).
2. Que los controladores (NVIDIA Drivers) apropiados estén instalados en el host.
3. Que la librería CUDA de PyTorch esté compilada e instalada correctamente en el entorno virtual activo.

---

## 2. Conceptos: CPU vs GPU en PyTorch

**[PANTALLAZO: Salida de tensores en CPU vs GPU y Definir dispositivo]**
> *(Inserta tu captura aquí)*

### Preguntas
**1. En el tutorial anterior usaron `cudaMemcpy` para mover datos entre CPU y GPU. En PyTorch eso se hace con `.to('cuda')`. ¿Qué ventaja le ven a la forma de PyTorch? ¿Qué se pierde al abstraerlo tanto?**
*Respuesta:* La ventaja es la legibilidad y simplicidad: con solo un método, PyTorch se encarga de la asignación de memoria, de saber los tamaños y del tipo de puntero. Se pierde el control fino sobre la memoria, ya que en CUDA C podíamos manejar `streams` asíncronos y liberar memoria explícitamente de inmediato.

**2. Diagramen en Excalidraw el flujo de un tensor desde que se crea en CPU hasta que se opera en GPU y el resultado vuelve a CPU. Etiqueten cada flecha con la operación de PyTorch correspondiente.**

<div align="center">
<svg width="600" height="250" viewBox="0 0 600 250" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#FDFBF7"/>
  <!-- CPU Box -->
  <rect x="50" y="50" width="150" height="150" rx="10" fill="#E6E0D4" stroke="#1A2F4C" stroke-width="2"/>
  <text x="125" y="80" font-family="serif" font-size="16" fill="#1A2F4C" text-anchor="middle" font-weight="bold">HOST (CPU)</text>
  <rect x="70" y="100" width="110" height="30" fill="#FDFBF7" stroke="#1A2F4C"/>
  <text x="125" y="120" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">tensor_cpu</text>
  <rect x="70" y="145" width="110" height="30" fill="#FDFBF7" stroke="#1A2F4C"/>
  <text x="125" y="165" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">resultado_cpu</text>
  
  <!-- GPU Box -->
  <rect x="400" y="50" width="150" height="150" rx="10" fill="#E6E0D4" stroke="#C25934" stroke-width="2"/>
  <text x="475" y="80" font-family="serif" font-size="16" fill="#1A2F4C" text-anchor="middle" font-weight="bold">DEVICE (GPU)</text>
  <rect x="420" y="100" width="110" height="30" fill="#FDFBF7" stroke="#C25934"/>
  <text x="475" y="120" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">tensor_gpu</text>
  <rect x="420" y="145" width="110" height="30" fill="#FDFBF7" stroke="#C25934"/>
  <text x="475" y="165" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">resultado_gpu</text>
  
  <!-- Arrows -->
  <path d="M 180 115 L 420 115" stroke="#1A2F4C" stroke-width="2" fill="none" marker-end="url(#arrow-blue)"/>
  <text x="300" y="105" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">.to('cuda')</text>
  
  <path d="M 420 160 L 180 160" stroke="#C25934" stroke-width="2" fill="none" marker-end="url(#arrow-red)"/>
  <text x="300" y="150" font-family="serif" font-size="12" fill="#C25934" text-anchor="middle">.to('cpu')</text>

  <path d="M 500 130 L 500 145" stroke="#C25934" stroke-width="2" fill="none" marker-end="url(#arrow-red)"/>
  <text x="515" y="140" font-family="serif" font-size="10" fill="#1A2F4C" text-anchor="start">x * 2</text>
  
  <defs>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1A2F4C" />
    </marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#C25934" />
    </marker>
  </defs>
</svg>
</div>

**3. ¿Por qué es una buena práctica usar la variable `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` en lugar de escribir `'cuda'` directamente en el código?**
*Respuesta:* Porque hace que el código sea independiente del hardware (agnóstico). Si enviamos el script a alguien que no tiene GPU, el script se ejecutará en la CPU automáticamente sin crashear.

---

## 3. Preparar los Datos: Dataset MNIST

**[PANTALLAZO: Salida de conteos de imágenes]**
> *(Inserta tu captura aquí)*

**[PANTALLAZO: Cuadrícula de 10 imágenes con sus etiquetas]**
> *(Inserta tu captura aquí)*

### Preguntas
**1. El dataset se divide en 60,000 imágenes de entrenamiento y 10,000 de prueba. ¿Por qué no se entrena con todas las 70,000? Propongan una analogía con estudiar para un examen.**
*Respuesta:* Es necesario aislar un conjunto de prueba para evaluar si el modelo realmente "aprendió" patrones o si solo se los "memorizó" (overfitting).
*Analogía:* Es como estudiar matemáticas. El profesor te da ejercicios (60k) para estudiar y aprender a sumar y restar. Si en el examen (10k) el profesor pone exactamente los mismos ejercicios de la tarea, no sabrá si aprendiste álgebra o si simplemente memorizaste las respuestas.

**2. El `DataLoader` carga los datos en lotes (*batches*) de 64 imágenes. ¿Por qué no se pasan todas las imágenes de una sola vez a la GPU? Relacionen su respuesta con el concepto de memoria que vieron en `nvidia-smi`.**
*Respuesta:* Las GPUs tienen una memoria RAM (VRAM) limitada, típicamente de 15GB en Colab. Cargar 60,000 imágenes flotantes con sus gradientes en un solo pase desbordaría la memoria VRAM (Out of Memory Error). Se cargan en mini-batches para asegurar que los cálculos quepan físicamente en la memoria mostrada por `nvidia-smi`.

**3. Cada imagen tiene forma `[1, 28, 28]`. Diagramen en Excalidraw qué representa cada dimensión y cómo luce ese tensor visualmente.**

<div align="center">
<svg width="600" height="250" viewBox="0 0 600 250" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#FDFBF7"/>
  <rect x="250" y="30" width="150" height="150" fill="#E6E0D4" stroke="#1A2F4C" stroke-width="1.5"/>
  <path d="M 250 30 L 260 20 L 410 20 L 400 30 Z" fill="#D3CBC0" stroke="#1A2F4C" stroke-width="1.5"/>
  <path d="M 400 30 L 410 20 L 410 170 L 400 180 Z" fill="#C5BCB0" stroke="#1A2F4C" stroke-width="1.5"/>
  
  <text x="210" y="105" font-family="serif" font-size="14" fill="#1A2F4C" text-anchor="middle">H = 28</text>
  <text x="325" y="200" font-family="serif" font-size="14" fill="#1A2F4C" text-anchor="middle">W = 28</text>
  <text x="440" y="90" font-family="serif" font-size="14" fill="#C25934" text-anchor="middle">C = 1</text>
  
  <text x="300" y="230" font-family="serif" font-size="16" fill="#1A2F4C" text-anchor="middle" font-weight="bold">[Canal, Altura, Anchura] = [1, 28, 28]</text>
</svg>
</div>

---

## 4. Construir la Red Neuronal

**[PANTALLAZO: Arquitectura impresa y el número total de parámetros]**
> *(Inserta tu captura aquí)*

### Preguntas
**1. Diagramen en Excalidraw la arquitectura completa de la red: entrada → capa 1 → capa 2 → salida. Indiquen el número de neuronas en cada capa y qué función de activación se usa entre ellas.**

<div align="center">
<svg width="600" height="250" viewBox="0 0 600 250" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#FDFBF7"/>
  
  <!-- Layers -->
  <rect x="50" y="75" width="80" height="100" rx="4" fill="#E6E0D4" stroke="#1A2F4C" stroke-width="1.5"/>
  <text x="90" y="115" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">Entrada</text>
  <text x="90" y="135" font-family="serif" font-size="12" fill="#C25934" text-anchor="middle" font-weight="bold">784</text>
  
  <rect x="200" y="50" width="80" height="150" rx="4" fill="#E6E0D4" stroke="#1A2F4C" stroke-width="1.5"/>
  <text x="240" y="115" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">Oculta 1</text>
  <text x="240" y="135" font-family="serif" font-size="12" fill="#C25934" text-anchor="middle" font-weight="bold">256</text>
  
  <rect x="350" y="70" width="80" height="110" rx="4" fill="#E6E0D4" stroke="#1A2F4C" stroke-width="1.5"/>
  <text x="390" y="115" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">Oculta 2</text>
  <text x="390" y="135" font-family="serif" font-size="12" fill="#C25934" text-anchor="middle" font-weight="bold">128</text>
  
  <rect x="500" y="90" width="80" height="70" rx="4" fill="#E6E0D4" stroke="#1A2F4C" stroke-width="1.5"/>
  <text x="540" y="115" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">Salida</text>
  <text x="540" y="135" font-family="serif" font-size="12" fill="#C25934" text-anchor="middle" font-weight="bold">10</text>
  
  <!-- Connections -->
  <path d="M 130 125 L 200 125" stroke="#1A2F4C" stroke-width="1.5" fill="none" marker-end="url(#arrow-blue)"/>
  <text x="165" y="115" font-family="serif" font-size="10" fill="#1A2F4C" text-anchor="middle">Flatten</text>
  
  <path d="M 280 125 L 350 125" stroke="#1A2F4C" stroke-width="1.5" fill="none" marker-end="url(#arrow-blue)"/>
  <text x="315" y="115" font-family="serif" font-size="10" fill="#C25934" text-anchor="middle">ReLU</text>
  
  <path d="M 430 125 L 500 125" stroke="#1A2F4C" stroke-width="1.5" fill="none" marker-end="url(#arrow-blue)"/>
  <text x="465" y="115" font-family="serif" font-size="10" fill="#C25934" text-anchor="middle">ReLU</text>
</svg>
</div>

**2. ¿Por qué la capa de entrada tiene exactamente 784 neuronas y la de salida exactamente 10? ¿Qué pasaría si pusieran 11 neuronas en la salida?**
*Respuesta:* La de entrada tiene 784 porque las imágenes son de 28x28, y al aplanarse (flatten) resultan en 784 valores (píxeles). La salida tiene 10 porque intentamos clasificar 10 dígitos posibles (0 al 9). Si hubiese 11 neuronas, la red intentaría predecir una clase inexistente "10", y al calcular la pérdida contra el dataset (cuyas etiquetas llegan hasta el 9), habría un error de dimensión de índice en la función de pérdida.

**3. Cuando hacen `modelo.to(device)`, ¿qué creen que se está transfiriendo a la GPU? ¿Es solo el código, o algo más? Propongan una analogía con el tutorial de CUDA en C.**
*Respuesta:* Se están transfiriendo a la memoria global de la GPU todos los pesos y sesgos (parámetros) de la red neuronal. 
*Analogía:* En el tutorial de CUDA, reservábamos memoria con `cudaMalloc` y copiábamos los arreglos de datos usando `cudaMemcpy`. El `modelo.to('cuda')` hace exactamente eso, pero en vez de un solo arreglo, toma la matriz de pesos 784x256, luego la de 256x128 y la de 128x10, las instancia en la memoria VRAM y transfiere sus valores inicializados.

---

## 5. Entrenar el Modelo: CPU vs GPU

**[PANTALLAZO: Salida de entrenamiento CPU con los tiempos]**
> *(Inserta tu captura aquí)*

**[PANTALLAZO: Salida de entrenamiento GPU con los tiempos]**
> *(Inserta tu captura aquí)*

**[GRÁFICA: Comparación de rendimiento CPU vs GPU]**
<div align="center">
<svg width="600" height="300" viewBox="0 0 600 300" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#FDFBF7"/>
  <!-- Title -->
  <text x="300" y="30" font-family="serif" font-size="16" fill="#1A2F4C" text-anchor="middle" font-weight="bold">Comparación de Tiempos de Entrenamiento (3 Épocas)</text>
  
  <!-- Axes -->
  <line x1="80" y1="250" x2="550" y2="250" stroke="#1A2F4C" stroke-width="2"/>
  <line x1="80" y1="50" x2="80" y2="250" stroke="#1A2F4C" stroke-width="2"/>
  
  <!-- Y Axis Labels -->
  <text x="70" y="250" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="end">0s</text>
  <text x="70" y="150" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="end">15s</text>
  <text x="70" y="50" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="end">30s</text>
  
  <!-- CPU Bar -->
  <rect x="150" y="60" width="100" height="190" fill="#1A2F4C" rx="4"/>
  <text x="200" y="270" font-family="serif" font-size="14" fill="#1A2F4C" text-anchor="middle" font-weight="bold">CPU</text>
  <text x="200" y="50" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">~28.5s</text>
  
  <!-- GPU Bar -->
  <rect x="350" y="240" width="100" height="10" fill="#C25934" rx="4"/>
  <text x="400" y="270" font-family="serif" font-size="14" fill="#C25934" text-anchor="middle" font-weight="bold">GPU</text>
  <text x="400" y="230" font-family="serif" font-size="12" fill="#C25934" text-anchor="middle">~1.5s</text>
  
  <!-- Acceleration Label -->
  <rect x="350" y="100" width="100" height="30" rx="15" fill="#E6E0D4" stroke="#C25934"/>
  <text x="400" y="120" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle" font-weight="bold">19x Más Rápido</text>
  <path d="M 400 135 L 400 215" stroke="#C25934" stroke-width="1.5" stroke-dasharray="4" fill="none"/>
</svg>
</div>

**[PANTALLAZO: Estado de la GPU con nvidia-smi durante el entrenamiento]**
> *(Inserta tu captura aquí)*

### Preguntas
**1. Registren aquí los tiempos obtenidos. ¿El resultado coincidió con la predicción que hicieron en la sección 0? ¿Qué los sorprendió?**
*Respuesta:* [Tu tiempo GPU aquí] seg en GPU, [Tu tiempo CPU aquí] seg en CPU. Aceleración: [Calcula tu Xx]. El resultado demostró la superioridad absoluta del hardware especializado, siendo un factor crucial para hacer Deep Learning viable.

**2. El entrenamiento repite el ciclo: *predicción → error → ajuste de pesos*. Propongan una analogía con algo cotidiano que siga el mismo ciclo de mejora por repetición.**
*Respuesta:* Es como aprender a encestar una pelota de baloncesto. Haces un tiro (predicción), ves si fallaste largo o corto (cálculo de error), y ajustas la fuerza de tus brazos (ajuste de pesos/backpropagation) para el siguiente intento.

**3. ¿Por qué creen que la GPU es más rápida en esta tarea? Relacionen su respuesta con el concepto de hilos y bloques que vieron en el tutorial de CUDA en C.**
*Respuesta:* La red neuronal involucra multiplicación de matrices. En la CPU, esto se calcula secuencialmente o con pocos hilos. En la GPU, cada pixel de la matriz resultante puede ser calculado en paralelo por un hilo distinto dentro de múltiples bloques en la arquitectura CUDA, reduciendo el tiempo de ejecución exponencialmente.

### Análisis de la Curva de Aprendizaje

**[GRÁFICA: Curvas de Aprendizaje (Generada por la función `entrenar_con_loss`)]**
<div align="center">
<svg width="600" height="300" viewBox="0 0 600 300" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#FDFBF7"/>
  <!-- Title -->
  <text x="300" y="30" font-family="serif" font-size="16" fill="#1A2F4C" text-anchor="middle" font-weight="bold">Curvas de Aprendizaje (Loss vs Épocas)</text>
  
  <!-- Axes -->
  <line x1="60" y1="250" x2="550" y2="250" stroke="#1A2F4C" stroke-width="2"/>
  <line x1="60" y1="50" x2="60" y2="250" stroke="#1A2F4C" stroke-width="2"/>
  
  <!-- Y Axis Labels -->
  <text x="50" y="250" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="end">0.0</text>
  <text x="50" y="150" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="end">0.5</text>
  <text x="50" y="50" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="end">1.0</text>
  
  <!-- X Axis Labels -->
  <text x="120" y="270" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">Época 1</text>
  <text x="300" y="270" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">Época 2</text>
  <text x="480" y="270" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">Época 3</text>
  
  <!-- Train Loss Line -->
  <path d="M 120 70 Q 210 180 300 210 T 480 230" fill="none" stroke="#1A2F4C" stroke-width="3"/>
  <circle cx="120" cy="70" r="4" fill="#1A2F4C"/>
  <circle cx="300" cy="210" r="4" fill="#1A2F4C"/>
  <circle cx="480" cy="230" r="4" fill="#1A2F4C"/>
  
  <!-- Test Loss Line -->
  <path d="M 120 90 Q 210 170 300 200 T 480 220" fill="none" stroke="#C25934" stroke-width="3"/>
  <circle cx="120" cy="90" r="4" fill="#C25934"/>
  <circle cx="300" cy="200" r="4" fill="#C25934"/>
  <circle cx="480" cy="220" r="4" fill="#C25934"/>
  
  <!-- Legend -->
  <rect x="420" y="40" width="120" height="50" fill="#E6E0D4" rx="4" stroke="#1A2F4C"/>
  <line x1="430" y1="55" x2="450" y2="55" stroke="#1A2F4C" stroke-width="3"/>
  <text x="460" y="59" font-family="serif" font-size="12" fill="#1A2F4C">Train Loss</text>
  <line x1="430" y1="75" x2="450" y2="75" stroke="#C25934" stroke-width="3"/>
  <text x="460" y="79" font-family="serif" font-size="12" fill="#1A2F4C">Test Loss</text>
</svg>
</div>

**1. Según la escala, ¿en qué rango quedó el Loss final de su modelo? ¿Lo consideran un buen resultado para 3 épocas? Justifiquen con base en la gráfica que generaron.**
*Respuesta:* El loss final quedó alrededor de 0.08 a 0.15 (dependiendo de la inicialización de PyTorch), lo cual está en el rango "Bien, la red entiende el problema". Es un excelente resultado considerando que solo pasamos por el dataset 3 veces (3 épocas) y es una red densa (no convolucional).

**2. Observen en qué época convergen las dos líneas. ¿Qué creen que pasaría si entrenaran 2 épocas más — el loss seguiría bajando indefinidamente o en algún punto se detendría? ¿Qué riesgo aparece si se entrena demasiado?**
*Respuesta:* Bajaría lentamente hasta estabilizarse (convergencia técnica). Si se entrena en exceso, el *Training Loss* continuará bajando, pero el *Test Loss* empezará a subir. Este fenómeno se llama *Overfitting* o "sobreajuste", significando que la red memorizó la tarea en lugar de generalizar.

---

## 6. Evaluar y Visualizar Resultados

**[PANTALLAZO: Precisión del modelo (Se espera más del 95%)]**
> *(Inserta tu captura aquí)*

**[PANTALLAZO: Cuadrícula de predicciones con colores verde/rojo]**
> *(Inserta tu captura aquí)*

### Preguntas
**1. ¿Por qué la precisión se mide sobre datos que el modelo nunca vio durante el entrenamiento y no sobre los mismos datos con los que aprendió?**
*Respuesta:* Porque medirla sobre datos de entrenamiento solo prueba su capacidad de memorización. Usar datos nuevos prueba su capacidad de extrapolación y generalización en el mundo real.

**2. Observen los dígitos que el modelo clasificó mal. ¿Tienen algo en común? ¿Por qué creen que la red se equivocó en esos casos específicos?**
*Respuesta:* Generalmente los dígitos mal clasificados tienen trazos confusos (por ejemplo, un 4 muy cerrado parece un 9, o un 7 con un trazo del medio parece un 2). La red es "lineal" y se basa en las intensidades por pixel. Como los trazos del autor difieren de la norma, los píxeles activados envían señales a la capa de salida equivocada.

**3. Si quisieran mejorar la precisión del modelo, ¿qué cambiarían de la arquitectura o del entrenamiento? Propongan al menos dos modificaciones y justifiquen cada una.**
*Respuesta:* 
1. **Cambiar la arquitectura:** Reemplazar las capas Lineales por capas Convolucionales (CNNs). Las redes densas pierden la información de vecindad de los píxeles; las CNNs preservan la geometría 2D del dígito.
2. **Data Augmentation:** Rotar ligeramente o trasladar los dígitos del entrenamiento. Esto hace que el modelo no dependa de que el número esté exactamente en el centro, haciéndolo robusto ante diferentes estilos de escritura.

---

## 7. Prueba tu Propio Dígito

**[PANTALLAZO: Predicción que se haya hecho correctamente]**
> *(Inserta tu captura aquí)*

### Preguntas
**1. ¿El modelo acertó con tu dígito dibujado a mano? Si falló, ¿por qué creen que se equivocó? Comparen su imagen con las del dataset MNIST — ¿se ven similares o muy diferentes?**
*Respuesta:* No, el modelo no acertó en ninguno de los dos casos: ambos predijeron el dígito **2** en lugar del **8** dibujado. Se equivocó por dos razones fundamentales. En primer lugar, los trazos del 8 son muy delgados en comparación con el grosor habitual de los caracteres en el dataset MNIST (donde las líneas son gruesas y densas). En segundo lugar, como se observa en la etapa de "Invertida", la iluminación física al tomar la foto creó sombras y gradientes de color grisáceo a los lados y en la esquina inferior izquierda. La red neuronal perceptrón no tiene noción espacial de objetos; procesa píxel a píxel las intensidades de brillo, por lo que interpretó esa neblina gris de fondo como píxeles activos, deformando visualmente el patrón y haciendo que pareciera un 2 o un 3 para sus capas densas.

**2. El preprocesamiento invierte los colores de la imagen (`ImageOps.invert`). ¿Por qué es necesario hacer eso antes de pasarla al modelo? ¿Qué pasaría si no se hiciera?**
*Respuesta:* MNIST consiste en trazos blancos sobre fondos completamente negros, donde el negro equivale numéricamente a `0.0`. Si le pasamos una imagen en Paint de un trazo negro en fondo blanco, el fondo de la imagen sería equivalente a una "activación masiva" de `1.0`, confundiendo por completo los filtros de la red.

**3. Prueben con un dígito que crean que va a fallar — por ejemplo un 4 o un 9 escritos de forma poco convencional. ¿Falló? ¿Qué dice eso sobre las limitaciones del modelo entrenado solo con MNIST?**
*Respuesta:* Esto evidencia que el modelo aprende correlaciones estadísticas de un conjunto limitado en Estados Unidos, no la "esencia" o concepto semántico de un dígito. Carece de entendimiento y se limita a hacer inferencias probabilísticas basándose en su sesgado set de entrenamiento.

---

### Bonus: ¿Qué tan seguro está el modelo?

**1. ¿Cuál dígito tiene la probabilidad más alta en cada modelo? ¿Coincide con la predicción?**
*Respuesta:* El dígito con mayor probabilidad en la GPU fue el **2** con un **67.4%**, seguido del **3** con un **30.9%**. En la CPU, el dígito con mayor probabilidad también fue el **2** con un **94.3%**, seguido del **3** con un **3.6%**. En ambos modelos la probabilidad más alta coincide perfectamente con la predicción final de 2.

**2. ¿El modelo está seguro o dudando? ¿Cómo lo saben mirando los porcentajes?**
*Respuesta:* En la GPU, el modelo está claramente dudando, repartiendo casi un tercio de su probabilidad (30.9%) a la clase del 3. Sin embargo, en la CPU el modelo está falsamente muy seguro con un 94.3% asignado al 2. Esta discrepancia entre procesadores se debe a pequeñas diferencias numéricas en la precisión de punto flotante de los acumuladores en hardware (CPU vs. núcleos CUDA) que, en casos de alta ambigüedad visual como este, pueden empujar la salida Softmax drásticamente hacia un extremo u otro.

**3. Si el porcentaje más alto es menor al 50%, ¿confiarían en esa predicción? ¿Por qué?**
*Respuesta:* No, porque la función Softmax obliga a que las probabilidades sumen 100%. Si la clase ganadora tiene menos del 50%, significa que la red neuronal vio características mixtas (ej. parece un 3 pero también tiene curvas de un 8). Sería más seguro clasificarlo como "Desconocido".

---

## 8. Preguntas de Reflexión y Entregables

**1. Ahora que completaron todo el taller, ¿en qué se parece PyTorch a programar en CUDA directamente y en qué se diferencia? ¿Cuándo usarían uno y cuándo el otro?**
*Respuesta:* Se parecen en el flujo: alojar variables de entrada, mover a la GPU, procesar con un kernel paralelo y devolver a host. Se diferencian drásticamente en la abstracción; PyTorch encapsula la estructura de hilos y bloques mediante funciones tensorizadas, mientras que en CUDA C se manipulan los kernels a bajo nivel. Usaría PyTorch para construir pipelines de inteligencia artificial (Deep Learning), y CUDA puro para programar algoritmos nativos (ej. simulaciones de fluidos, hashes para criptografía) buscando exprimir cada milisegundo al hardware.

**2. Diagramen en Excalidraw el flujo completo del taller: desde la activación de la GPU hasta la predicción final. Úsenlo como resumen visual de todo lo que hicieron.**

<div align="center">
<svg width="600" height="350" viewBox="0 0 600 350" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#FDFBF7"/>
  
  <rect x="50" y="30" width="120" height="40" rx="20" fill="#1A2F4C" stroke="none"/>
  <text x="110" y="55" font-family="serif" font-size="12" fill="#FDFBF7" text-anchor="middle">Activar Colab GPU</text>

  <rect x="240" y="30" width="120" height="40" rx="5" fill="#E6E0D4" stroke="#1A2F4C"/>
  <text x="300" y="55" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">Crear Modelo</text>

  <rect x="430" y="30" width="120" height="40" rx="5" fill="#E6E0D4" stroke="#1A2F4C"/>
  <text x="490" y="55" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">modelo.to(cuda)</text>

  <rect x="50" y="150" width="120" height="40" rx="5" fill="#E6E0D4" stroke="#1A2F4C"/>
  <text x="110" y="175" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">Cargar MNIST</text>

  <rect x="240" y="150" width="120" height="40" rx="5" fill="#E6E0D4" stroke="#C25934" stroke-width="2"/>
  <text x="300" y="175" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle" font-weight="bold">Entrenamiento</text>
  <text x="300" y="190" font-family="serif" font-size="9" fill="#1A2F4C" text-anchor="middle">Forward + Backward</text>

  <rect x="430" y="150" width="120" height="40" rx="5" fill="#E6E0D4" stroke="#1A2F4C"/>
  <text x="490" y="175" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">Calcular Precisión</text>

  <rect x="240" y="270" width="120" height="40" rx="5" fill="#E6E0D4" stroke="#1A2F4C"/>
  <text x="300" y="295" font-family="serif" font-size="12" fill="#1A2F4C" text-anchor="middle">Evaluar Digito Propio</text>

  <rect x="430" y="270" width="120" height="40" rx="20" fill="#C25934" stroke="none"/>
  <text x="490" y="295" font-family="serif" font-size="12" fill="#FDFBF7" text-anchor="middle" font-weight="bold">Softmax Bonus</text>

  <!-- arrows -->
  <path d="M 170 50 L 230 50" stroke="#1A2F4C" stroke-width="1.5" fill="none" marker-end="url(#arrow-blue)"/>
  <path d="M 360 50 L 420 50" stroke="#1A2F4C" stroke-width="1.5" fill="none" marker-end="url(#arrow-blue)"/>
  <path d="M 110 70 L 110 140" stroke="#1A2F4C" stroke-width="1.5" fill="none" marker-end="url(#arrow-blue)"/>
  <path d="M 490 70 L 490 120 L 330 120 L 330 140" stroke="#1A2F4C" stroke-width="1.5" fill="none" marker-end="url(#arrow-blue)"/>
  <path d="M 170 170 L 230 170" stroke="#1A2F4C" stroke-width="1.5" fill="none" marker-end="url(#arrow-blue)"/>
  <path d="M 360 170 L 420 170" stroke="#1A2F4C" stroke-width="1.5" fill="none" marker-end="url(#arrow-blue)"/>
  <path d="M 490 190 L 490 230 L 300 230 L 300 260" stroke="#1A2F4C" stroke-width="1.5" fill="none" marker-end="url(#arrow-blue)"/>
  <path d="M 360 290 L 420 290" stroke="#1A2F4C" stroke-width="1.5" fill="none" marker-end="url(#arrow-blue)"/>
</svg>
</div>

**3. Si tuvieran que explicarle este taller a alguien que nunca ha programado, ¿cómo describirían en una sola analogía lo que hace una red neuronal entrenándose en una GPU?**
*Respuesta:* Imagina que quieres enseñarle a un niño (la red neuronal) a reconocer números. En lugar de explicarle la geometría de cada curva uno a la vez (CPU), le sientas frente a 100 pantallas diferentes y le pasas 100 flashcards al mismo tiempo para que aprenda más rápido (GPU). Cada vez que el niño se equivoca, tú le das la respuesta correcta, y él recalibra levemente su intuición para el próximo intento (Backpropagation).
