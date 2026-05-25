# Entrenamiento de Redes Neuronales en GPU
* CUDA con PyTorch en Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Albonire/parcial-gpu-pytorch-colab/blob/main/Taller_CUDA_PyTorch.ipynb)

---

| | |
|---|---|
| **Parcial** | Segundo Corte |
| **Materia** | Programación Paralela y Computación Distribuida |
| **Profesor** | Prf. Juan Alejandro Carrillo Jaimes |
| **Integrantes** | Anderson González |
| | |
| **Fecha** | 2026-I |

---

## 0. Instrucciones Generales
* El taller se desarrolla en Google Colab usando una GPU gratuita de NVIDIA.
* Se trabaja en parejas; ambos integrantes deben entender cada parte.
* Se deben capturar pantallazos de cada salida importante indicada con [PANTALLAZO].
* Al finalizar, se descarga el notebook y se sube todo a un repositorio de GitHub.

### Preguntas
**1. ¿Qué diferencia hay entre un notebook en la nube (Colab) y un entorno local como el del tutorial de instalación? ¿Cuál prefieren y por qué?**
*Respuesta:* [Tu respuesta aquí]

**2. Antes de comenzar, hagan una predicción: ¿cuántas veces más rápida creen que será la GPU comparada con la CPU en el entrenamiento? Anoten su predicción aquí y compárenla al final con el resultado real.**
*Respuesta:* [Tu predicción aquí]

---

## 1. Configurar el Entorno en Google Colab
* Activar la GPU desde el menú de Colab: Entorno de ejecución > Cambiar tipo de entorno de ejecución.
* Verificar que PyTorch reconoce la GPU y mostrar el nombre del dispositivo.
* Ejecutar `nvidia-smi` para ver el estado de la GPU, igual que en el tutorial de instalación.

**[PANTALLAZO: Verificar GPU disponible (Debe mostrar: GPU disponible: True y el nombre de la GPU)]**
> *(Inserta tu captura aquí)*

**[PANTALLAZO: Estado de la GPU con nvidia-smi (Identificar: nombre de la GPU, versión de CUDA, memoria total)]**
> *(Inserta tu captura aquí)*

### Preguntas
**1. La salida de `nvidia-smi` muestra campos como *Driver Version*, *Memory Usage* y *GPU-Util*. ¿Qué indica cada uno?**
*Respuesta:* [Tu respuesta aquí]

**2. Cuando activan el acelerador en Colab, ¿qué creen que ocurre físicamente? ¿La GPU está en su computador o en otro lugar? Propongan una analogía con algo de la vida cotidiana.**
*Respuesta:* [Tu respuesta aquí]

**3. `torch.cuda.is_available()` retorna `True` o `False`. ¿Qué condiciones deben cumplirse para que retorne `True`? Listen al menos tres requisitos.**
*Respuesta:* [Tu respuesta aquí]

---

## 2. Conceptos: CPU vs GPU en PyTorch
* Comparar las operaciones de CUDA en C con su equivalente en PyTorch.
* Entender cómo se mueven tensores entre CPU y GPU con `.to('cuda')`.
* Definir el dispositivo al inicio del proyecto para que el código funcione con o sin GPU.

**[PANTALLAZO: Salida de tensores en CPU vs GPU y Definir dispositivo (Confirmen que tensor_gpu.device dice cuda:0)]**
> *(Inserta tu captura aquí)*

### Preguntas
**1. En el tutorial anterior usaron `cudaMemcpy` para mover datos entre CPU y GPU. En PyTorch eso se hace con `.to('cuda')`. ¿Qué ventaja le ven a la forma de PyTorch? ¿Qué se pierde al abstraerlo tanto?**
*Respuesta:* [Tu respuesta aquí]

**2. Diagramen en Excalidraw el flujo de un tensor desde que se crea en CPU hasta que se opera en GPU y el resultado vuelve a CPU. Etiqueten cada flecha con la operación de PyTorch correspondiente.**
*Respuesta:* [Inserta tu diagrama de Excalidraw aquí]

**3. ¿Por qué es una buena práctica usar la variable `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` en lugar de escribir `'cuda'` directamente en el código?**
*Respuesta:* [Tu respuesta aquí]

---

## 3. Preparar los Datos: Dataset MNIST
* Descargar el dataset MNIST: 60,000 imágenes de entrenamiento y 10,000 de prueba.
* Aplicar transformaciones para convertir las imágenes a tensores y normalizarlas.
* Visualizar una muestra del dataset para entender qué se va a clasificar.

**[PANTALLAZO: Salida de conteos de imágenes]**
> *(Inserta tu captura aquí)*

**[PANTALLAZO: Cuadrícula de 10 imágenes con sus etiquetas]**
> *(Inserta tu captura aquí)*

### Preguntas
**1. El dataset se divide en 60,000 imágenes de entrenamiento y 10,000 de prueba. ¿Por qué no se entrena con todas las 70,000? Propongan una analogía con estudiar para un examen.**
*Respuesta:* [Tu respuesta aquí]

**2. El `DataLoader` carga los datos en lotes (*batches*) de 64 imágenes. ¿Por qué no se pasan todas las imágenes de una sola vez a la GPU? Relacionen su respuesta con el concepto de memoria que vieron en `nvidia-smi`.**
*Respuesta:* [Tu respuesta aquí]

**3. Cada imagen tiene forma `[1, 28, 28]`. Diagramen en Excalidraw qué representa cada dimensión y cómo luce ese tensor visualmente.**
*Respuesta:* [Inserta tu diagrama de Excalidraw aquí]

---

## 4. Construir la Red Neuronal
* Definir la arquitectura: capa de entrada (784), dos capas ocultas (256 y 128), capa de salida (10 dígitos).
* Mover el modelo a la GPU con `.to(device)`.
* Contar el total de parámetros entrenables de la red.

**[PANTALLAZO: Arquitectura impresa y el número total de parámetros]**
> *(Inserta tu captura aquí)*

### Preguntas
**1. Diagramen en Excalidraw la arquitectura completa de la red: entrada → capa 1 → capa 2 → salida. Indiquen el número de neuronas en cada capa y qué función de activación se usa entre ellas.**
*Respuesta:* [Inserta tu diagrama de Excalidraw aquí]

**2. ¿Por qué la capa de entrada tiene exactamente 784 neuronas y la de salida exactamente 10? ¿Qué pasaría si pusieran 11 neuronas en la salida?**
*Respuesta:* [Tu respuesta aquí]

**3. Cuando hacen `modelo.to(device)`, ¿qué creen que se está transfiriendo a la GPU? ¿Es solo el código, o algo más? Propongan una analogía con el tutorial de CUDA en C.**
*Respuesta:* [Tu respuesta aquí]

---

## 5. Entrenar el Modelo: CPU vs GPU
* Entrenar el mismo modelo dos veces: primero en CPU, luego en GPU.
* Medir el tiempo de entrenamiento en cada dispositivo.
* Comparar los resultados y calcular cuántas veces más rápida fue la GPU.

**[PANTALLAZO: Salida de entrenamiento CPU con los tiempos]**
> *(Inserta tu captura aquí)*

**[PANTALLAZO: Salida de entrenamiento GPU con los tiempos]**
> *(Inserta tu captura aquí)*

**[PANTALLAZO: Comparación de rendimiento (incluyendo nvidia-smi)]**
> *(Inserta tu captura aquí)*

### Preguntas
**1. Registren aquí los tiempos obtenidos. ¿El resultado coincidió con la predicción que hicieron en la sección 0? ¿Qué los sorprendió?**
*Respuesta:* [Tu respuesta aquí]

**2. El entrenamiento repite el ciclo: *predicción → error → ajuste de pesos*. Propongan una analogía con algo cotidiano que siga el mismo ciclo de mejora por repetición.**
*Respuesta:* [Tu respuesta aquí]

**3. ¿Por qué creen que la GPU es más rápida en esta tarea? Relacionen su respuesta con el concepto de hilos y bloques que vieron en el tutorial de CUDA en C.**
*Respuesta:* [Tu respuesta aquí]

### Análisis de la Curva de Aprendizaje

**1. Según la escala, ¿en qué rango quedó el Loss final de su modelo? ¿Lo consideran un buen resultado para 3 épocas? Justifiquen con base en la gráfica que generaron.**
*Respuesta:* [Tu respuesta aquí]

**2. Observen en qué época convergen las dos líneas. ¿Qué creen que pasaría si entrenaran 2 épocas más — el loss seguiría bajando indefinidamente o en algún punto se detendría? ¿Qué riesgo aparece si se entrena demasiado?**
*Respuesta:* [Tu respuesta aquí]

---

## 6. Evaluar y Visualizar Resultados
* Calcular la precisión del modelo sobre los datos de prueba que nunca vio durante el entrenamiento.
* Visualizar predicciones reales con indicadores de acierto (verde) y error (rojo).

**[PANTALLAZO: Precisión del modelo (Se espera más del 95%)]**
> *(Inserta tu captura aquí)*

**[PANTALLAZO: Cuadrícula de predicciones con colores verde/rojo]**
> *(Inserta tu captura aquí)*

### Preguntas
**1. ¿Por qué la precisión se mide sobre datos que el modelo nunca vio durante el entrenamiento y no sobre los mismos datos con los que aprendió?**
*Respuesta:* [Tu respuesta aquí]

**2. Observen los dígitos que el modelo clasificó mal. ¿Tienen algo en común? ¿Por qué creen que la red se equivocó en esos casos específicos?**
*Respuesta:* [Tu respuesta aquí]

**3. Si quisieran mejorar la precisión del modelo, ¿qué cambiarían de la arquitectura o del entrenamiento? Propongan al menos dos modificaciones y justifiquen cada una.**
*Respuesta:* [Tu respuesta aquí]

---

## 7. Prueba tu Propio Dígito
* Dibujar un dígito del 0 al 9 en Paint (o cualquier editor), guardarlo como imagen.
* Subir la imagen a Colab y preprocesarla para que tenga el mismo formato que MNIST: escala de grises, fondo negro, trazo blanco, tamaño 28x28.
* Pasarla al modelo entrenado y ver qué predice.
* Visualizar la imagen tal como la ve la red antes de hacer la predicción.

**[PANTALLAZO: Predicción que se haya hecho correctamente]**
> *(Inserta tu captura aquí)*

### Preguntas
**1. ¿El modelo acertó con tu dígito dibujado a mano? Si falló, ¿por qué creen que se equivocó? Comparen su imagen con las del dataset MNIST — ¿se ven similares o muy diferentes?**
*Respuesta:* [Tu respuesta aquí]

**2. El preprocesamiento invierte los colores de la imagen (`ImageOps.invert`). ¿Por qué es necesario hacer eso antes de pasarla al modelo? ¿Qué pasaría si no se hiciera?**
*Respuesta:* [Tu respuesta aquí]

**3. Prueben con un dígito que crean que va a fallar — por ejemplo un 4 o un 9 escritos de forma poco convencional. ¿Falló? ¿Qué dice eso sobre las limitaciones del modelo entrenado solo con MNIST?**
*Respuesta:* [Tu respuesta aquí]

---

### Bonus: ¿Qué tan seguro está el modelo?
Hasta ahora sabemos *qué* predice el modelo, pero no *qué tan seguro* está de su respuesta. Un modelo puede predecir "7" con un 95% de confianza o con un 40% — y eso hace toda la diferencia.

**Observen y respondan:**
**1. ¿Cuál dígito tiene la probabilidad más alta en cada modelo? ¿Coincide con la predicción?**
*Respuesta:* [Tu respuesta aquí]

**2. ¿El modelo está seguro o dudando? ¿Cómo lo saben mirando los porcentajes?**
*Respuesta:* [Tu respuesta aquí]

**3. Si el porcentaje más alto es menor al 50%, ¿confiarían en esa predicción? ¿Por qué?**
*Respuesta:* [Tu respuesta aquí]

---

## 8. Preguntas de Reflexión y Entregables
* Responder 4 preguntas que conectan lo aprendido en PyTorch con el tutorial de CUDA en C.
* Subir a GitHub el notebook descargado y un reporte en Markdown con pantallazos y respuestas.

### Preguntas
**1. Ahora que completaron todo el taller, ¿en qué se parece PyTorch a programar en CUDA directamente y en qué se diferencia? ¿Cuándo usarían uno y cuándo el otro?**
*Respuesta:* [Tu respuesta aquí]

**2. Diagramen en Excalidraw el flujo completo del taller: desde la activación de la GPU hasta la predicción final. Úsenlo como resumen visual de todo lo que hicieron.**
*Respuesta:* [Inserta tu diagrama de Excalidraw aquí]

**3. Si tuvieran que explicarle este taller a alguien que nunca ha programado, ¿cómo describirían en una sola analogía lo que hace una red neuronal entrenándose en una GPU?**
*Respuesta:* [Tu respuesta aquí]
